function GenerateMultiRAGQuery_Cerebras_Qwen3_32B(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Cerebras_Qwen3_32B
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    /no_think

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Groq_Qwen3_32B(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Groq_Qwen3_32B
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Cerebras_GPT_OSS_120B_Medium(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Cerebras_GPT_OSS_120B_MEDIUM_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    /no_think

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Groq_GPT_OSS_120B_Medium(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Groq_GPT_OSS_120B_MEDIUM_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Baseten_GPT_OSS_120B_Medium(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Baseten_GPT_OSS_120B_MEDIUM_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    Output Format:
    {{ ctx.output_format }}
    "#
}


function GenerateMultiRAGQuery_Cerebras_GPT_OSS_120B_Low(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Cerebras_GPT_OSS_120B_LOW_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    /no_think

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Groq_GPT_OSS_120B_Low(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Groq_GPT_OSS_120B_LOW_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    Output Format:
    {{ ctx.output_format }}
    "#
}

function GenerateMultiRAGQuery_Baseten_GPT_OSS_120B_Low(
    conversation_history: ConversationMessage[],
    user_query: string
) -> MultiRAGQueryResult {
    client Baseten_GPT_OSS_120B_LOW_REASONING
    prompt #"
    You are a Multi-Query RAG Generation Agent. Analyze the full conversation context to generate 2-3 focused search queries addressing the user's underlying information needs.

    KEY APPROACH:
    1. Analyze the COMPLETE conversation thread to identify the main topic being discussed
    2. Generate queries about the actual information need, not communication issues

    PRINCIPLES:
    - Generate all queries in Korean language (한국어)
    - Focus on substantive topics from conversation analysis
    - Avoid queries about communication problems - target the underlying subject matter

    Context:
    - User's Latest Query: {{ user_query }}
    - Conversation History: {{ conversation_history }}

    Generate 2-3 Korean search queries that address the user's actual information need based on the conversation context.

    Output Format:
    {{ ctx.output_format }}
    "#
}
