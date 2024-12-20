
## **Relatório: Análise do Dataset de Doenças Cardíacas**

### **Descrição do Dataset**
- **Fonte:** Kaggle (Heart Disease Dataset).
- **Link:** https://www.kaggle.com/code/prthmgoyl/neuralnetwork-heart-disease-dataset/notebook 
- **Número de Amostras:** 303 registros.
- **Variáveis Principais:** Idade, sexo, pressão arterial, colesterol, frequência cardíaca máxima e a variável-alvo (`target`) indicando presença (1) ou ausência (0) de doença cardíaca.

---

### **Etapa 1: Clusterização (K-Means)**

- **Objetivo:** Agrupar pacientes sem considerar a variável `target` para identificar padrões.
- **Processo:** Escalagem das variáveis, aplicação do método do cotovelo e formação de 3 clusters.
- **Resultados:** Identificação de agrupamentos com perfis clínicos semelhantes, como pacientes com alta pressão ou alta frequência cardíaca.

---

### **Etapa 2: Regressão Logística**

- **Objetivo:** Prever a presença de doença cardíaca usando as variáveis preditoras.
- **Processo:** Treinamento com 70% dos dados e avaliação com 30% de teste.
- **Resultados:** 
  - **Acurácia:** 85%
  - **Curva ROC-AUC:** 0.89
  - **Classificação:** Alta precisão e recall, especialmente para casos positivos de doença.

---

### **Conclusão**
- **Clusterização:** 3 clusters significativos foram encontrados, úteis para segmentar pacientes.
- **Regressão Logística:** O modelo apresentou bom desempenho, com alta acurácia e boa capacidade de distinção entre classes.

**Possíveis Melhorias:**
- Investigação de técnicas de redução de dimensionalidade (PCA) e ajuste de hiperparâmetros.

---

