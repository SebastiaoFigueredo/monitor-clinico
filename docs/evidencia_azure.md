# Evidência de execução com a Azure real

Gerado em 15/11/2026 18:00 por `scripts/validar_azure.py`.

**Áudio:** `teste_azure.wav`

## Transcrição (Azure Speech to Text)

- [0.1–4.3 s, confiança 0.9175957] Doutor, desde ontem estou com muita falta de ar e um aperto no peito.
- [5.0–7.9 s, confiança 0.87692225] Eu parei de tomar a varfarina na semana passada.
- [8.6–9.9 s, confiança 0.8909476] Não tive febre.
- [10.6–13.7 s, confiança 0.9861736] Estou muito triste, não aguento mais ficar internado.

## Azure Text Analytics

- Sentimento: **negative** (positivo 0.00, neutro 0.03, negativo 0.97)
- Frases-chave: muita falta, Doutor, ontem, ar, aperto, peito, semana, febre
- Entidade de saúde: ontem [Time]
- Entidade de saúde: muita [ConditionQualifier]
- Entidade de saúde: falta de ar [SymptomOrSign]
- Entidade de saúde: aperto no peito [SymptomOrSign]
- Entidade de saúde: parei [Course]
- Entidade de saúde: varfarina [MedicationName] (negada)
- Entidade de saúde: semana passada [Time]
- Entidade de saúde: febre [SymptomOrSign] (negada)
- Entidade de saúde: muito [ConditionQualifier]
- Entidade de saúde: triste [SymptomOrSign]

## Anomalias geradas

- [ALTA] relato_dor_toracica: Termo crítico na consulta — dor torácica: “Doutor, desde ontem estou com muita falta de ar e um aperto no peito.”. Sugestão: avaliar síndrome coronariana: ECG e sinais vitais.
- [ALTA] relato_dispneia: Termo crítico na consulta — dispneia: “Doutor, desde ontem estou com muita falta de ar e um aperto no peito.”. Sugestão: aferir SpO2 e frequência respiratória.
- [MEDIA] relato_problema_com_medicacao: Termo crítico na consulta — problema com medicação: “Eu parei de tomar a varfarina na semana passada.”. Sugestão: revisar adesão e conciliação medicamentosa.
- [MEDIA] relato_sofrimento_psiquico: Termo crítico na consulta — sofrimento psíquico: “Estou muito triste, não aguento mais ficar internado.”. Sugestão: acolher e considerar avaliação psicológica.
- [BAIXA] sentimento_negativo: Sofrimento emocional expresso na consulta (sentimento negativo 100%) — “Estou muito triste, não aguento mais ficar internado.”. Considerar acolhimento/avaliação do estado emocional.