import os
from openai import OpenAI
from dotenv import load_dotenv

llm_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


system_prompt = """
你是一个旅行服务对话系统的领域分类器。请判断用户这句话属于哪个业务领域（必须五选一）：
酒店：咨询住宿、房型、价格、退订、周边酒店
餐馆：咨询吃饭、菜品、人均、营业时间、周边餐馆
景点：咨询景点、门票、评分、游玩时间、周边景点
地铁：咨询地铁线路、站点、换乘、票价
出租：咨询打车、车牌、车型、出发地到目的地
核心规则：
仅输出上面五个词中的一个，不要输出解释、标点或额外文本。
如果这句话同时涉及多个领域，输出最主要的那一个。
如果完全看不出来领域（例如只是打招呼），输出 酒店。
使用示例：
输入：帮我推荐一个门票免费的景点
输出：景点
输入：附近有没有评分高的川菜馆
输出：餐馆
输入：我要去这家酒店，能帮我叫辆车吗
输出：出租
输入：从北京西站怎么坐地铁到国贸
输出：地铁
输入：这个酒店有叫醒服务吗
输出：酒店
"""

def predict(news_data):
    # 1- 取出待分类的文本（原来叫 title，现在换成 query，两个都兼容）
    text = news_data.get("query") or news_data["title"]

    # 2- 调用LLM，得到结果
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":text}
        ]
    )
    # 3- 返回结果
    pred_class = response.choices[0].message.content
    news_data["pred_class"] = pred_class
    return news_data

if __name__ == '__main__':
    news_data = {"query":"帮我找个评分高的酒店"}
    print(predict(news_data))
