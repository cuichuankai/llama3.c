#include "mongoose.h"
#include "server_mode.h"
#include <stdbool.h>
#include <time.h>

struct Transformer;
struct Tokenizer;
struct Sampler;

float *forward(struct Transformer *transformer, int token, int pos);
int sample(struct Sampler *sampler, float *logits);
void encode(struct Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char *decode(struct Tokenizer *t, int prev_token, int token);

struct server_ctx {
  struct Transformer *transformer;
  struct Tokenizer *tokenizer;
  struct Sampler *sampler;
  int steps;
  const char *model_name;
  const char *model_id;
  time_t start_ts;
};

static void log_rotate_if_needed(FILE *f, const char *path) {
  long size = 0;
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  if (size > 10485760) {
    fclose(f);
    remove("server.log.1");
    rename(path, "server.log.1");
    FILE *nf = fopen(path, "w");
    if (nf) fclose(nf);
  }
}

static void server_log(const char *text) {
  FILE *f = fopen("server.log", "a");
  if (!f) return;
  fprintf(f, "%s\n", text);
  log_rotate_if_needed(f, "server.log");
  fclose(f);
}

static char *str_dup_n(const char *s, size_t n) {
  char *r = (char *)malloc(n + 1);
  if (!r) return NULL;
  memcpy(r, s, n);
  r[n] = '\0';
  return r;
}

struct msg_item { char *role; char *content; };

static int parse_messages(struct mg_str body, struct msg_item **out_items) {
  struct mg_str arr = mg_json_get_tok(body, "$.messages");
  if (arr.buf == NULL) return 0;
  size_t ofs = 0;
  int count = 0;
  struct msg_item *items = NULL;
  while ((ofs = mg_json_next(arr, ofs, NULL, NULL)) > 0) count++;
  if (count == 0) { *out_items = NULL; return 0; }
  items = (struct msg_item *)calloc((size_t)count, sizeof(struct msg_item));
  ofs = 0;
  int idx = 0;
  while ((ofs = mg_json_next(arr, ofs, NULL, NULL)) > 0) {
    char path_role[64];
    char path_content[64];
    snprintf(path_role, sizeof(path_role), "$.messages[%d].role", idx);
    snprintf(path_content, sizeof(path_content), "$.messages[%d].content", idx);
    char *role = mg_json_get_str(body, path_role);
    char *content = mg_json_get_str(body, path_content);
    items[idx].role = role ? role : str_dup_n("user", 4);
    items[idx].content = content ? content : str_dup_n("", 0);
    idx++;
  }
  *out_items = items;
  return count;
}

static void free_messages(struct msg_item *items, int n) {
  for (int i = 0; i < n; i++) {
    free(items[i].role);
    free(items[i].content);
  }
  free(items);
}

static char *run_chat(struct server_ctx *ctx, struct msg_item *items, int n, int steps_override) {
  int steps = steps_override > 0 ? steps_override : ctx->steps;
  int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *tmp_tokens = (int *)malloc(32768 * sizeof(int));
  int num_prompt_tokens = 0;
  int *system_tokens = (int *)malloc(32768 * sizeof(int));
  int num_system_tokens = 0;
  int have_system = 0;
  for (int i = 0; i < n; i++) {
    if (items[i].role && strcmp(items[i].role, "system") == 0) have_system = 1;
  }
  if (have_system) {
    prompt_tokens[num_prompt_tokens++] = 128000;
    prompt_tokens[num_prompt_tokens++] = 128006;
    prompt_tokens[num_prompt_tokens++] = 9125;
    prompt_tokens[num_prompt_tokens++] = 128007;
    prompt_tokens[num_prompt_tokens++] = 271;
    for (int i = 0; i < n; i++) {
      if (items[i].role && strcmp(items[i].role, "system") == 0) {
        num_system_tokens = 0;
        encode(ctx->tokenizer, items[i].content, 0, 0, system_tokens, &num_system_tokens);
        for (int j = 0; j < num_system_tokens; j++) prompt_tokens[num_prompt_tokens++] = system_tokens[j];
      }
    }
    prompt_tokens[num_prompt_tokens++] = 128009;
  }
  for (int i = 0; i < n; i++) {
    const char *role = items[i].role ? items[i].role : "user";
    int role_id = 882;
    if (strcmp(role, "assistant") == 0) role_id = 78191;
    if (strcmp(role, "system") == 0) role_id = 9125;
    prompt_tokens[num_prompt_tokens++] = 128006;
    prompt_tokens[num_prompt_tokens++] = role_id;
    prompt_tokens[num_prompt_tokens++] = 128007;
    prompt_tokens[num_prompt_tokens++] = 271;
    int num_tmp = 0;
    encode(ctx->tokenizer, items[i].content, 0, 0, tmp_tokens, &num_tmp);
    for (int j = 0; j < num_tmp; j++) prompt_tokens[num_prompt_tokens++] = tmp_tokens[j];
    prompt_tokens[num_prompt_tokens++] = 128009;
  }
  prompt_tokens[num_prompt_tokens++] = 128006;
  prompt_tokens[num_prompt_tokens++] = 78191;
  prompt_tokens[num_prompt_tokens++] = 128007;
  prompt_tokens[num_prompt_tokens++] = 271;
  int user_idx = 0;
  int next = 0;
  int token = 0;
  int pos = 0;
  size_t out_cap = 8192;
  size_t out_len = 0;
  char *out = (char *)malloc(out_cap);
  if (!out) out = str_dup_n("", 0);
  while (pos < steps) {
    if (user_idx < num_prompt_tokens) token = prompt_tokens[user_idx++]; else token = next;
    float *logits = forward(ctx->transformer, token, pos);
    next = sample(ctx->sampler, logits);
    pos++;
    if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
      char *piece = decode(ctx->tokenizer, token, next);
      size_t plen = piece ? strlen(piece) : 0;
      if (out_len + plen + 1 > out_cap) {
        out_cap = (out_len + plen + 1) * 2;
        char *tmp = (char *)realloc(out, out_cap);
        if (tmp) out = tmp;
      }
      if (piece) {
        memcpy(out + out_len, piece, plen);
        out_len += plen;
        out[out_len] = '\0';
      }
    }
    if (user_idx >= num_prompt_tokens && (next == 128009 || next == 128001)) break;
  }
  free(prompt_tokens);
  free(tmp_tokens);
  free(system_tokens);
  return out;
}


static char *json_escape(const char *s) {
  if (!s) return str_dup_n("", 0);
  size_t n = strlen(s);
  size_t cap = n * 2 + 1;
  char *r = (char *)malloc(cap);
  size_t j = 0;
  for (size_t i = 0; i < n; i++) {
    char ch = s[i];
    if (ch == '"' || ch == '\\') { r[j++] = '\\'; r[j++] = ch; }
    else if (ch == '\n') { r[j++] = '\\'; r[j++] = 'n'; }
    else if (ch == '\r') { r[j++] = '\\'; r[j++] = 'r'; }
    else { r[j++] = ch; }
  }
  r[j] = '\0';
  return r;
}

static void handle_chat(struct mg_connection *c, struct mg_http_message *hm, struct server_ctx *ctx) {
  struct msg_item *items = NULL;
  int n = parse_messages(hm->body, &items);
  int steps_override = mg_json_get_long(hm->body, "$.max_tokens", 0);
  bool stream_flag = false;
  mg_json_get_bool(hm->body, "$.stream", &stream_flag);
  if (n <= 0) {
    mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"error\":{\"message\":\"messages required\"}}");
    return;
  }
  if (stream_flag) {
    mg_printf(c, "HTTP/1.1 200 OK\r\n");
    mg_printf(c, "Content-Type: text/event-stream\r\n");
    mg_printf(c, "Cache-Control: no-cache\r\n");
    mg_printf(c, "Connection: keep-alive\r\n");
    mg_printf(c, "Transfer-Encoding: chunked\r\n\r\n");
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int *tmp_tokens = (int *)malloc(32768 * sizeof(int));
    int num_prompt_tokens = 0;
    int *system_tokens = (int *)malloc(32768 * sizeof(int));
    int num_system_tokens = 0;
    int have_system = 0;
    for (int i = 0; i < n; i++) {
      if (items[i].role && strcmp(items[i].role, "system") == 0) have_system = 1;
    }
    if (have_system) {
      prompt_tokens[num_prompt_tokens++] = 128000;
      prompt_tokens[num_prompt_tokens++] = 128006;
      prompt_tokens[num_prompt_tokens++] = 9125;
      prompt_tokens[num_prompt_tokens++] = 128007;
      prompt_tokens[num_prompt_tokens++] = 271;
      for (int i = 0; i < n; i++) {
        if (items[i].role && strcmp(items[i].role, "system") == 0) {
          num_system_tokens = 0;
          encode(ctx->tokenizer, items[i].content, 0, 0, system_tokens, &num_system_tokens);
          for (int j = 0; j < num_system_tokens; j++) prompt_tokens[num_prompt_tokens++] = system_tokens[j];
        }
      }
      prompt_tokens[num_prompt_tokens++] = 128009;
    }
    for (int i = 0; i < n; i++) {
      const char *role = items[i].role ? items[i].role : "user";
      int role_id = 882;
      if (strcmp(role, "assistant") == 0) role_id = 78191;
      if (strcmp(role, "system") == 0) role_id = 9125;
      prompt_tokens[num_prompt_tokens++] = 128006;
      prompt_tokens[num_prompt_tokens++] = role_id;
      prompt_tokens[num_prompt_tokens++] = 128007;
      prompt_tokens[num_prompt_tokens++] = 271;
      int num_tmp = 0;
      encode(ctx->tokenizer, items[i].content, 0, 0, tmp_tokens, &num_tmp);
      for (int j = 0; j < num_tmp; j++) prompt_tokens[num_prompt_tokens++] = tmp_tokens[j];
      prompt_tokens[num_prompt_tokens++] = 128009;
    }
    prompt_tokens[num_prompt_tokens++] = 128006;
    prompt_tokens[num_prompt_tokens++] = 78191;
    prompt_tokens[num_prompt_tokens++] = 128007;
    prompt_tokens[num_prompt_tokens++] = 271;
    int user_idx = 0;
    int next = 0;
    int token = 0;
    int pos = 0;
    while (pos < (steps_override > 0 ? steps_override : ctx->steps)) {
      if (user_idx < num_prompt_tokens) token = prompt_tokens[user_idx++]; else token = next;
      float *logits = forward(ctx->transformer, token, pos);
      next = sample(ctx->sampler, logits);
      pos++;
      if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
        char *piece = decode(ctx->tokenizer, token, next);
        char *ep = json_escape(piece);
        time_t now = time(NULL);
        char hdr[256 + 1024];
        snprintf(hdr, sizeof(hdr), "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}", (long)now, ctx->model_id, ep);
        mg_http_printf_chunk(c, "data: %s\n\n", hdr);
        mg_mgr_poll(c->mgr, 0);
        free(ep);
      }
      if (user_idx >= num_prompt_tokens && (next == 128009 || next == 128001)) break;
    }
    mg_http_printf_chunk(c, "data: [DONE]\n\n");
    mg_mgr_poll(c->mgr, 0);
    mg_http_write_chunk(c, "", 0);
    free(prompt_tokens);
    free(tmp_tokens);
    free(system_tokens);
    free_messages(items, n);
  } else {
    char *content = run_chat(ctx, items, n, steps_override);
    free_messages(items, n);
    time_t now = time(NULL);
    char *escaped = content ? content : "";
    size_t elen = strlen(escaped);
    char *json = (char *)malloc(elen + 512);
    if (!json) json = str_dup_n("{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":\"stop\"}]}", 121);
    else snprintf(json, elen + 512, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}]}", (long)now, ctx->model_id, escaped);
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json);
    free(json);
    free(content);
  }
}

static void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
  struct server_ctx *ctx = (struct server_ctx *)c->fn_data;
  if (ev == MG_EV_HTTP_MSG) {
    struct mg_http_message *hm = (struct mg_http_message *)ev_data;
    struct mg_http_serve_opts opts = {0};
    opts.root_dir = "assets";
    if (mg_strcmp(hm->uri, mg_str("/v1/chat/completions")) == 0) {
      server_log("POST /v1/chat/completions");
      handle_chat(c, hm, ctx);
    } else if (mg_strcmp(hm->uri, mg_str("/healthz")) == 0) {
      mg_http_reply(c, 200, "Content-Type: text/plain\r\n", "ok");
    } else if (mg_strcmp(hm->uri, mg_str("/readyz")) == 0) {
      time_t now = time(NULL);
      long up = (long)(now - ctx->start_ts);
      char buf[256];
      snprintf(buf, sizeof(buf), "{\"status\":\"ready\",\"model\":\"%s\",\"uptime\":%ld}", ctx->model_id, up);
      mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", buf);
    } else if (mg_strcmp(hm->uri, mg_str("/")) == 0) {
      mg_http_serve_file(c, hm, "assets/index.html", &opts);
    } else if (mg_match(hm->uri, mg_str("/assets/**"), NULL)) {
      mg_http_serve_dir(c, hm, &opts);
    } else {
      mg_http_reply(c, 404, "Content-Type: text/plain\r\n", "Not found");
    }
  }
}

void server_mode(struct Transformer *transformer, struct Tokenizer *tokenizer, struct Sampler *sampler, int steps, const char *model_name) {
  struct server_ctx ctx;
  ctx.transformer = transformer;
  ctx.tokenizer = tokenizer;
  ctx.sampler = sampler;
  ctx.steps = steps;
  ctx.model_name = model_name;
  ctx.start_ts = time(NULL);
  const char *bn = model_name;
  const char *p = model_name;
  while (*p) { if (*p == '/' || *p == '\\') bn = p + 1; p++; }
  ctx.model_id = bn;
  struct mg_mgr mgr;
  mg_mgr_init(&mgr);
  server_log("starting server");
  const char *port_env = getenv("LLAMA3_PORT");
  if (!port_env) port_env = getenv("PORT");
  char url[64];
  snprintf(url, sizeof(url), "http://0.0.0.0:%s", port_env ? port_env : "8000");
  struct mg_connection *lc = mg_http_listen(&mgr, url, ev_handler, &ctx);
  if (lc == NULL) {
    server_log("listen failed");
    return;
  }
  for (;;) mg_mgr_poll(&mgr, 100);
  mg_mgr_free(&mgr);
}
