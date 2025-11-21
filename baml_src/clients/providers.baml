client<llm> Cerebras_Qwen3_32B {
  provider "openai-generic"
  options {
    base_url "https://api.cerebras.ai/v1"
    api_key env.CEREBRAS_API_KEY
    model "qwen-3-32b" 
    temperature 0.0
  }
}

client<llm> Cerebras_GPT_OSS_120B_MEDIUM_REASONING {
  provider "openai-generic"
  options {
    base_url "https://api.cerebras.ai/v1"
    api_key env.CEREBRAS_API_KEY
    model "gpt-oss-120b"
    temperature 0.0
    reasoning_effort "medium"
  }
}

client<llm> Cerebras_GPT_OSS_120B_LOW_REASONING {
  provider "openai-generic"
  options {
    base_url "https://api.cerebras.ai/v1"
    api_key env.CEREBRAS_API_KEY
    model "gpt-oss-120b"
    temperature 0.0
    reasoning_effort "low"
  }
}

client<llm> Groq_Qwen3_32B {
  provider "openai-generic"
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "qwen/qwen3-32b" 
    temperature 0.0
  }
}


client<llm> Groq_GPT_OSS_120B_MEDIUM_REASONING {
  provider openai-generic
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "openai/gpt-oss-120b"
    temperature 0.0
    reasoning_effort "medium"
  }
}

client<llm> Groq_GPT_OSS_120B_LOW_REASONING {
  provider openai-generic
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "openai/gpt-oss-120b"
    temperature 0.0
    reasoning_effort "low"
  }
}