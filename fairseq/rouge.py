from rouge import Rouge


class RougeScorer:
   def __init__(self):
      """
      Initializes `Rouge` scorer. 
      """
      self.scorer = Rouge()
      self.hypothesis = []
      self.references = []

   def add_string(self, hypothesis, reference):
      """
      Adds the given hypothesis and reference sentences
      to evaluate them later.
      :param hypothesis: The hypothesis sentence.
      :param reference: The reference sentence.
      """
      self.hypothesis.append(hypothesis)
      self.references.append(reference)

   def result_string(self):
      """
      Calculates the `Rouge` score of every hypothesis-reference
      pair.
      """
      return self.scorer.get_scores(self.hypothesis, self.references, avg=True)