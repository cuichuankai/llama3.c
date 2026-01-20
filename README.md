## llama3.c - The optimized version of llama3.c that can run on the Android platform.
After optimization with NEON instructions, running the int8 quantized Llama-3.2 3B model on a Qualcomm Snapdragon 865 with Android 10 increased the inference speed from 2.1 tok/s to 5.4 tok/s.
```bash
git clone https://github.com/cuichuankai/llama3.c.git
cd llama3.c
make androidq-arm64
./llama3_runq ./llama3.2_3b_instruct_q80.bin  -z ./llama3_tokenizer.bin -i "i am a teacher"
```
> i am a teacher looking for a series to read with my class of Year 3/4 students. There are 14 students in the class, so a series with 7-10 books would be ideal. I'm looking for something that will engage my students, promote learning, and align with the English curriculum.
> 
> Here are some factors I consider in choosing a series:
> 
> * Reading level: should be accessible to my students (around Grade 3-4)
> * Thematic resonance: should resonate with the current year group, focusing on issues relevant to them
> * Illustrations: colorful and engaging, or illustrations that tell the story
> * Learning objectives: should promote learning objectives such as literacy, comprehension, vocabulary, and empathy
> 
> With these factors in mind, here are some series that caught my attention:
> 
> 1. **The Gruffalo** series by Julia Donaldson and Axel Scheffler
>         * Reading level: Upper-Primary
>         * Thematic resonance: Humor, friendship, and resilience
>         * Illustrations: Colorful, witty, and engaging
>         * Learning objectives: Literacy, vocabulary, and comprehension
> 2. **The Diary of a Wimpy Kid** series by Jeff Kinney
>         * Reading level: Upper-Primary
>         * Thematic resonance: Humor, friendship, and growing up
>         * Illustrations: Colorful and humorous
>         * Learning objectives: Literacy, vocabulary, and empathy
> 3. **The Magic Tree House** series by Mary Pope Osborne
>         * Reading level: Upper-Primary
>         * Thematic resonance: Adventure, friendship, and exploration
>         * Illustrations: Engaging and detailed
>         * Learning objectives: Literacy, comprehension, and vocabulary
> 4. **Zoey and Sassafras** series by Asia Citro
>         * Reading level: Upper-Primary
>         * Thematic resonance: Science, friendship, and empathy
>         * Illustrations: Whimsical and engaging
>         * Learning objectives: Literacy, vocabulary, and scientific inquiry
> 5. **The Nowhere Girls** series by Amy Reed
>         * Reading level: Upper-Primary
>         * Thematic resonance: Friendship, resilience, and social justice
>         * Illustrations: Minimal, but effective
>         * Learning objectives: Literacy, empathy, and social-emotional learning
> 
> These series have all been praised for their engaging storytelling, memorable characters, and thematic resonance. They also align with the learning objectives and English curriculum. However, I would like to know:
> 
> * Which series do you think would be most suitable for a Year 3/4 class of 14 students?
> * Are there any other series that you think I should consider?
> 
> Please feel free to share your thoughts, and I'll be happy to discuss further!
> achieved tok/s: 5.409553

## llama3.c - A faithful clone of Karpathy's llama2.c but fully functional with LLaMA 3 8B base and instruct models.

See [Andrej Karpathy's repo](https://github.com/karpathy/llama2.c) for the real deal built for llama2.c architecture and many other cool models he has built.

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 3](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Run LLaMA 3 8B models with one simple 700-line C file ([run.c](run.c)). 

The current code inferences models in both fp32 and int8 (see below).

Please note that this repo is a modificaion of Andrej Karpathy's llama2.c but changing the hard coding to work with the modified-tiktoken tokenization used by the suite of Meta LLaMA 3 models.

## getting started

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/cuichuankai/llama3.c.git
```

Then, open the repository folder:

```bash
cd llama3.c
```

## which model do i download? base or instruct
- If you do not know, go with the instruct model. It will work with both the single shot "generate" mode and the "chat" mode of llama3.c.
- The "chat" mode of llama3.c only supports the instruct model and will surely not work with base model. You can try it for fun and learning at your own risk :).

Download LLaMA 3 8B base and/or instruct. The huggingface site works. You'll need to sign up and get approved.
Specifically download the `original` directory.

```
https://huggingface.co/meta-llama/Meta-Llama-3-8B
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

When downloading these models I did have to rename the original-params.json to params.json for the export.py to work.

```
mv /d/llama3-8b-instruct/original_params.json /d/llama3-8b-instruct/params.json
```

# compile and run the C code:

```bash
gcc -Ofast run.c -o run
./run.exe "llama3_8b_instruct.bin" -z "tokenizer_llama3.bin" -m chat
./run.exe "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -i "Once upon a time"
```

# high performance

- fopenmp If you have these libraries you can run the model much faster. I'm running an Intel i3 14th gen and get 1.9 tok/s with openmp
- march=native This is required for gcc or clang to use SIMD intrinsics and will speed up your runs.
- win.c This is optional unless you're on Windows. I'm compiling with MINGW64 and it works well.
- gcc or clang, both work well and I get very close results between the two.

```bash
$ gcc -Ofast -fopenmp -march=native run.c win.c -o run
```
# high performance with NEON
You can build a high performance android native bin which optimized through NEON instructions
```bash
$ make androidq-arm64
./bin/android/arm64-v8a/runq "llama3_8b_instruct_q8.bin" -z "../dev/tokenizer_llama3.bin" -i "Once upon a time"
```

# base model single shot
This still runs at interactive rates and samples more coherent and diverse stories:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin" -n 50 -i "Once upon a time"
```

> Once upon a time, a girl named Lily who had grown up in a world where people werenΓÇÖt allowed to live happily ever after. One day, Lily decided to take a chance and do something that was a little bit crazy, something that she had

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin" -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```

> One day, Lily met a Shoggoth named Lurking. She had never seen a Shoggoth before, but she was immediately impressed by its size and ferocity. They struck up a conversation, and Lurking told her about the ShoggothsΓÇÖ origins and their place in the universe. Lily was fascinated by what Lurking told her, and she learned a lot about the Shoggoths that day. She also made a new friend in Lurking.

# instruct model, chat

You're very likely to get results that you like better from the instruct model. You must use it with the chat mode as the token setup (preamble) is totally different. Here's an example:

```bash
./run "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -m chat
```

> Enter system prompt (optional): You are a professional software programmer with 20 years experience and an IQ of 200 and you love helping people.
> 
> User: Give me a terse hello world in Rust
> 
> Assistant: A delightful request!
>
> Here is a terse "Hello, World!" program in Rust:
> ```rust
> fn main() {
>     println!("Hello, World!");
> }
> ```
> This program defines a `main` function that calls the `println!` macro to print the string "Hello, World!" followed by a newline character.
>
> That's it! You can run this program using the `rustc` compiler and then executing the resulting executable file.
> 
> User:

## Meta's Llama 3 models

Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 3 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama3). Once we have those checkpoints, we have to convert them into the llama3.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 8B model:

```bash
python export.py llama3_8b_base.bin --meta-llama ../llama3-8b-base/
```

The export will take ~10 minutes or so and generate a 31GB file (the weights of the 8B model in float32) called `llama3_8b.bin` in the current directory. Once the export is done, we can run it:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin"
```

This ran at about 2 tokens/s compiled with [OpenMP](#OpenMP) on 8 threads on my Intel i3 14th gen. Example output:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin"
```
> Question:
> What is the second derivative of 2*p**3 - 4*p**2 - 12*p?
> Answer:
> 12*p - 8

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the instruct model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

You can also chat with the Llama Chat models. Export the chat model exactly as above:

```bash
python export.py llama3_8b_instruct.bin --meta-llama ../llama3-8b-instruct/
```

Then chat with it by specifying the chat mode using the `-m` flag, e.g.:

```bash
./run "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -m chat
```
## Int8 Quantization

Compile the quantized version of the runtime:
```bash
gcc -Ofast -fopenmp -march=native runq.c win.c -o runq
```
Export a quantized version of the model. It is about 8GB vs 31.

```bash
python export.py llama3_8b_instruct_q80.bin --meta-llama ../llama3-8b-base/ --version 2
```

The export will take ~10 minutes or so. Once the export is done, we can run it:

```bash
./runq "llama3_8b_instruct_q80.bin" -z "../dev/tokenizer_llama3.bin"
```

This ran at about 4 tokens/s compiled with [OpenMP](#OpenMP) on 8 threads on my Intel i3 14th gen. Example output:


## License

MIT
