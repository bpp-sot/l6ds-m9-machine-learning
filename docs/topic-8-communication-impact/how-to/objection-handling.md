# Anticipate & Handle Objections

> Do not get defensive. Objections are signs of engagement.

## Common Archetypes

### "The Black Box Skeptic"
* **Objection:** "How do we know why it chose to deny this loan? We can't use a black box."
* **Handling:** Introduce local interpretability immediately. "That's exactly why we use LIME. For every single denial, the model outputs a receipt showing the top 3 reasons, like 'Income too low' or 'Recent late payment'. Would you like to see an example receipt?"

### "The Perfect-or-Nothing Believer"
* **Objection:** "You said the accuracy is 85%. That means it's wrong 15% of the time. We can't risk that."
* **Handling:** Anchor against the current baseline. "You're right, it is not perfect. However, our manual human review process is currently running at 65% accuracy. This model represents a 20% absolute improvement, saving 40 hours of manual work a week."

### "The Edge-Case Finder"
* **Objection:** "What if a customer has a spelling mistake in their name, lives abroad, and uses a VPN? Will it break?"
* **Handling:** Acknowledge and redirect. "Excellent edge case. We haven't explicitly trained for that exact combination. However, our fallback protocol automatically routes low-confidence predictions (under 50%) to a human agent. The model doesn't handle everything, it handles the 80% routine traffic."

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S5 | Deployment, value assessment, and ROI | Translating model performance into business impact |
| S6 | Communicate through storytelling and visualisation | Presenting ML results to non-technical stakeholders |
| B4 | Consideration of organisational goals | Framing technical results in terms of business objectives |
| B1 | Inquisitive approach | Exploring creative ways to explain model behaviour |
