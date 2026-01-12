# Annotation Guidelines: Sentiment Analysis

This document provides clear guidelines for annotating text sentiment.
Consistent annotations are crucial for building reliable machine learning models.

## General Principles

1. **Be Consistent**: Apply the same criteria to all texts
2. **Consider Context**: Understand the full meaning before labeling
3. **Don't Overthink**: Trust your first impression for obvious cases
4. **When in Doubt**: Mark as neutral or flag for review

## Labels

### Positive

**Definition:** The text expresses satisfaction, happiness, approval, or positive emotions about the subject.

**Key Indicators:**
- Praise or compliments
- Expressions of satisfaction
- Enthusiastic language
- Positive outcomes mentioned

**Examples:**
- "This product exceeded my expectations!"
- "Excellent service, highly recommend!"
- "I love this! Best purchase ever."
- "Great quality, works perfectly."

### Negative

**Definition:** The text expresses dissatisfaction, disappointment, disapproval, or negative emotions about the subject.

**Key Indicators:**
- Complaints or criticism
- Expressions of disappointment
- Negative outcomes mentioned
- Warning others

**Examples:**
- "Terrible quality, waste of money."
- "Very disappointed with this purchase."
- "Broke after one day, do not buy."
- "Worst experience ever, horrible service."

### Neutral

**Definition:** The text is balanced, factual, lacks clear emotion, or contains equal positive and negative elements.

**Key Indicators:**
- Factual statements without emotion
- Balanced reviews mentioning pros and cons equally
- Questions or requests for information
- Objective descriptions

**Examples:**
- "It's okay, nothing special."
- "Received the product as described."
- "Average quality, average price."
- "Has some good features but also some drawbacks."

## Edge Cases and How to Handle Them

### Mixed Sentiment
**Case:** Text contains both positive and negative aspects.

**How to Handle:**
- Identify the overall dominant sentiment
- If truly balanced, label as neutral
- Example: "Good quality but too expensive" → Often neutral or slight negative depending on emphasis

### Sarcasm and Irony
**Case:** Text says something positive but means negative (or vice versa).

**How to Handle:**
- Label based on the intended meaning, not literal words
- Example: "Oh great, another broken product" → Negative

### Questions
**Case:** Text is primarily a question.

**How to Handle:**
- Label as neutral unless the question clearly expresses emotion
- "Does this work?" → Neutral
- "Why is this so terrible?" → Negative

### Comparisons
**Case:** Text compares to other products.

**How to Handle:**
- Focus on the sentiment about the current product
- "Better than Brand X" → Positive
- "Not as good as I expected" → Negative

### Spam or Irrelevant
**Case:** Text is not a genuine review (spam, gibberish, etc.).

**How to Handle:**
- Skip or mark for removal
- Do not try to assign sentiment to spam

## Common Pitfalls to Avoid

### ❌ DON'T:
- Let your personal opinions influence labels
- Label based on topic rather than sentiment
- Change your criteria mid-annotation
- Rush through without reading completely

### ✓ DO:
- Read the entire text before deciding
- Consider the author's perspective
- Apply guidelines consistently
- Take breaks to maintain quality
- Flag unclear cases for review

## Quality Checklist

Before finalizing your annotations, ask yourself:

- [ ] Did I read the entire text?
- [ ] Is my label based on the text content, not my opinion?
- [ ] Am I being consistent with previous similar texts?
- [ ] Would another annotator likely choose the same label?
- [ ] Did I handle edge cases according to guidelines?

## Examples by Difficulty

### Easy (Clear sentiment)
✓ "Amazing product! Love it!" → Positive
✓ "Complete waste of money." → Negative  
✓ "It arrived on time." → Neutral

### Medium (Requires careful reading)
✓ "Not the worst, but could be better." → Negative (leaning)
✓ "Has potential but needs improvements." → Neutral or Negative
✓ "Surprisingly good for the price." → Positive

### Hard (Ambiguous cases)
✓ "I guess it's fine." → Neutral (lukewarm)
✓ "It works, nothing more nothing less." → Neutral
✓ "Expected more but not terrible." → Negative or Neutral (context dependent)

## Getting Help

If you encounter:
- **Unclear text**: Flag for review or skip
- **Technical terms you don't understand**: Research or flag
- **Disagree with example**: Discuss with team lead
- **Fatigue**: Take a break to maintain quality

Remember: Quality > Quantity. It's better to annotate fewer items correctly than many items incorrectly.
