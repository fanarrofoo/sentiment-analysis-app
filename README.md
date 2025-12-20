## 🧪 Model Selection & Research
I conducted a comprehensive benchmarking study to identify the optimal architecture for this 7-class sentiment classification task. Despite the popularity of Deep Learning, traditional statistical methods proved most effective for this specific dataset.

### 📊 Comparative Results

| Model Architecture | Category | F1-Score | Inference Speed | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **LinearSVC + TF-IDF** | **Statistical** | **0.8604** | **Instant** | **🏆 Champion** |
| BERT (Base) | Transformer (Encoder) | 0.6600 | Slow (GPU req.) | Overfit/Noisy |
| FastText (Tuned) | Embeddings | 0.5211 | Fast | Data Hungry |
| GPT-2 (Fine-tuned) | Transformer (Decoder) | 0.4322 | Very Slow | Causal/Generative |

### 🔍 Key Findings & Engineering Insights

1. **Simplicity Wins:** The **LinearSVC** model paired with **TF-IDF (N-grams 1,2)** outperformed modern Transformers by over 20%. This suggests the dataset contains high-impact "anchor words" that statistical weighting identifies more precisely than deep learning attention mechanisms.
2. **Context vs. Noise:** While BERT and GPT-2 are powerful, their high parameter count (100M+) led to overfitting on this specific 7-class distribution. The noise introduced by the generative nature of GPT-2 was particularly detrimental to classification accuracy.
3. **Efficiency:** The winning model is 10,000x smaller in file size than BERT and runs instantly on a standard CPU, making it the most sustainable and cost-effective choice for production deployment.
