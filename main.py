from model import *
from vocabulary  import *
from visualization import *
from train import *


if __name__ == "__main__":
    # train("data\zh.txt")
    test_zh_words=["政府","国家","部门","发展","增长","战略","体系","调整","党","学校","政策"]
    test_en_words=["government","nation","department","country","growth","strategy","system","adjustment","party","school","policy"]
    
    vis=Visualization("data\zh.txt","embeddings_zh_10.pth")
    vis.find_similar_words("总统",top_k=5)
    vis=Visualization("data\en.txt","embeddings_en_10.pth")
    vis.find_similar_words("nation",top_k=5)

