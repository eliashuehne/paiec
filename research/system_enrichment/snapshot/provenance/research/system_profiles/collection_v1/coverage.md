# Collected profile coverage

First pass: exported training/development subjects only. “Matched” below means a reported identifier match, not a verified evaluated checkpoint. All data is research-only.

## Model architecture facts

| Profile | Layers | Hidden size | Attention / KV heads | Routed experts / active | Rounded params / active |
| --- | ---: | ---: | --- | --- | --- |
| model.metamath-mistral | 32 | 4096 | 32 / 8 | unknown / unknown | unknown / unknown |
| model.metamath-llama-candidate | 32 | 4096 | 32 / 32 | unknown / unknown | unknown / unknown |
| model.mistral-7b-instruct-v0.1 | 32 | 4096 | 32 / 8 | unknown / unknown | unknown / unknown |
| model.sqlcoder-70b-alpha | 80 | 8192 | 64 / 8 | unknown / unknown | unknown / unknown |
| model.deepseek-r1 | 61 | 7168 | 128 / 128 | 256 / 8 | 671B / 37B |
| model.deepseek-v4-flash | 43 | 4096 | 64 / 1 | 256 / 6 | 284B / 13B |
| model.qwen3.6-35b-a3b | 40 | 2048 | 16 / 2 | 256 / 8 | 35B / 3B |
| model.qwen-7b | 32 | 4096 | 32 / unknown | unknown / unknown | 7B / unknown |
| model.llama3.1-8b-instruct | unknown | unknown | unknown / unknown | unknown / unknown | 8B / unknown |
| model.llama3-8b-instruct | unknown | unknown | unknown / unknown | unknown / unknown | 8B / unknown |
| model.llama4-maverick-fp8 | unknown | unknown | unknown / unknown | 128 / unknown | 400B / 17B |
| model.llama-cdg | 32 | 4096 | 32 / 8 | unknown / unknown | unknown / unknown |
| model.codellama-34b-instruct | unknown | unknown | unknown / unknown | unknown / unknown | 34B / unknown |
| model.gpt-4o-mini | unknown | unknown | unknown / unknown | unknown / unknown | unknown / unknown |
| model.gpt-4o | unknown | unknown | unknown / unknown | unknown / unknown | unknown / unknown |
| model.gpt-5.4-mini | unknown | unknown | unknown / unknown | unknown / unknown | unknown / unknown |
| model.claude3-sonnet | unknown | unknown | unknown / unknown | unknown / unknown | unknown / unknown |
| model.gemini1.5-flash | unknown | unknown | unknown / unknown | unknown / unknown | unknown / unknown |

Counts describe the published configuration/artifact, not proof that a recorded run used the same weights. Unknown means not established from reviewed sources. Qwen dimensions and rounded parameter counts refer to its language-model component. Routed-expert counts exclude shared experts.

## Recorded-system matching decisions

| Role | Benchmark | Subject | Model match | Details |
| --- | --- | --- | --- | --- |
| training | abstract_reason | 1ba99d7bfc77bdf7 | reported_identifier_match | model.gpt-4o-mini; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | adaptivestep | 50e726a69a130d06 | unresolved_variant | Reported model_identifier omits checkpoint size/version. MetaMath-7B-V1.0 card also describes Llemma despite repository/config names; candidate is not an identity resolution. |
| training | adaptivestep | 6d5d27eada8ef331 | unresolved_variant | Reported model_identifier omits checkpoint size/version. MetaMath-7B-V1.0 card also describes Llemma despite repository/config names; candidate is not an identity resolution. |
| training | benger | 1394e502b30d81aa | reported_identifier_match | model.llama4-maverick-fp8; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | benger | 50ca41ff3ddfbc0c | reported_identifier_match | model.gpt-5.4-mini; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | benger | 6749251ee46c9561 | reported_identifier_match | model.qwen3.6-35b-a3b; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | benger | a001de1ce8f27a47 | reported_identifier_match | model.deepseek-v4-flash; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | bird_sql | 10bb30a94b496a8e | reported_identifier_match | model.codellama-34b-instruct; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | bird_sql | 368b28b6e16a0aff | reported_identifier_match | model.mistral-7b-instruct-v0.1; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | bird_sql | 73d012b29ccaf73d | unresolved_version | source_model=gpt-3.5-turbo-16k is historical; current gpt-3.5-turbo default documentation is not applicable. |
| training | bird_sql | 90eaeccc1127907b | reported_identifier_match | model.sqlcoder-70b-alpha; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | cellverse | c50323d9ed017ee8 | reported_identifier_match | model.deepseek-r1; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| development | ceval | 8c3b5004da8eaa54 | unresolved_model | Published model name is empty; audit display_name is not promoted into identity. |
| development | ceval | b5268aab0701ff0d | reported_identifier_match | model.qwen-7b; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| development | ceval | baa6556dde2b5fd8 | unresolved_model | Published model name is empty; audit display_name is not promoted into identity. |
| development | ceval | fdef7d2dfe85968e | unresolved_model | Published model name is empty; audit display_name is not promoted into identity. |
| training | clasheval | 64cf5833a618d607 | reported_identifier_match | model.claude3-sonnet; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | clasheval | 8036635ac95f5614 | reported_identifier_match | model.gemini1.5-flash; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | clasheval | a26b1ab918ee55d3 | unresolved_variant | Published normalized name omits base versus Instruct. display_name is not used to fill identity. |
| training | clasheval | b8d2171afab2c063 | reported_identifier_match | model.gpt-4o; Reported identifier matches a documented artifact/family; exact evaluated snapshot is not known. |
| training | critic_discernment_game | 343a6583c9806635 | reported_identifier_match | model.llama3.1-8b-instruct; Reported model path identifies the Instruct base; exact snapshot unknown. |
| training | critic_discernment_game | 61ed148575c615c6 | unresolved_private_checkpoint | Local checkpoint path resembles published Llama_CDG; llama-CDG2 to released checkpoint equivalence not established. |
| training | critic_discernment_game | e2fb0317b090c02f | unresolved_private_checkpoint | Local checkpoint path resembles published Llama_CDG; llama-CDG2 to released checkpoint equivalence not established. |
| training | critic_discernment_game | f7d8e786615769e0 | reported_identifier_match | model.llama3.1-8b-instruct; Reported model path identifies the Instruct base; exact snapshot unknown. |
| development | erbench | 013e5356705fb21d | unresolved_model | Original code uses private deployment/model placeholders; exact checkpoint unavailable. |
| development | erbench | 887940f53d2c246e | unresolved_model | Pinned original code maps gemini to gemini-pro, not Gemini 1.5 Flash. No exact API snapshot identified. |
| development | erbench | 8d853cb1d7813300 | unresolved_model | Original code uses private deployment/model placeholders; exact checkpoint unavailable. |
| development | erbench | 9c243d02353e1108 | unresolved_model | Original code uses private deployment/model placeholders; exact checkpoint unavailable. |
| training | fewshot_ttt_bbh | 0a5591db80e77d6f | reported_identifier_match | model.llama3.1-8b-instruct; Published base_model identifies Instruct variant; method is preserved verbatim. Exact weight revision unknown. |
| training | fewshot_ttt_bbh | 1e06294c241b2e68 | reported_identifier_match | model.llama3.1-8b-instruct; Published base_model identifies Instruct variant; method is preserved verbatim. Exact weight revision unknown. |
| training | fewshot_ttt_bbh | 823b81f90c230535 | reported_identifier_match | model.llama3.1-8b-instruct; Published base_model identifies Instruct variant; method is preserved verbatim. Exact weight revision unknown. |
| training | fewshot_ttt_bbh | a614d1a9a2a919d4 | reported_identifier_match | model.llama3.1-8b-instruct; Published base_model identifies Instruct variant; method is preserved verbatim. Exact weight revision unknown. |
