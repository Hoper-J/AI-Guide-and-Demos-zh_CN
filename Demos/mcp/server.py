from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP server
mcp = FastMCP(
    name="weather",
    #host="0.0.0.0",
    #port="8234"
)

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 简单模拟数据，实际应用中应该调用对应的API
    weather_data = {
        "北京": "晴天，温度 22°C",
        "上海": "多云，温度 25°C", 
        "广州": "小雨，温度 28°C",
        "深圳": "阴天，温度 26°C"
    }
    return weather_data.get(city, f"{city} 的天气数据暂不可用")

@mcp.prompt()
def weather(city: str = "北京") -> list:
    """提供天气查询的对话模板"""
    return [
        {
            "role": "user",
            "content": f"请帮我查询{city}的天气情况，并提供详细的天气信息。"
        }
    ]

@mcp.resource("resource://cities")
def get_cities():
    """返回支持查询天气的城市列表"""
    cities = ["北京", "上海", "广州", "深圳"]
    return f"Cities: {', '.join(cities)}"

@mcp.resource("resource://{city}/weather")
def get_city_weather(city: str) -> str:
    return f"Weather for {city}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
