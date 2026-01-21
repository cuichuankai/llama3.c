#ifndef SERVER_MODE_H
#define SERVER_MODE_H

struct Transformer;
struct Tokenizer;
struct Sampler;

void server_mode(struct Transformer *transformer, struct Tokenizer *tokenizer, struct Sampler *sampler, int steps, const char *model_name);

#endif
