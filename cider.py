from cider_scorer import CiderScorer

class Cider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, n=4, sigma=6.0, lang='en'):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self._lang = lang

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        assert(gts.keys() == res.keys()), (len(gts.keys()), len(res.keys()))
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma, lang=self._lang)

        for i in imgIds:
            hypo = res[i]
            ref = gts[i]

            # assert(type(hypo) is list)
            assert(type(hypo) is str)
            # assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo, ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

"""test
cider = Cider(lang='zh')
gts = {0:['这是个书'], 1:['你是一个好人']}
res = {0:'那是个书', 1:'我是一个好人'}
# gts = {0:['This is a test.'], 1:['I am a good person.']}
# res = {0:'This is a quiz.', 1:'You are a good person.'}
print(cider.compute_score(gts, res))
"""
