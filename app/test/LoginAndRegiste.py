import requests
import json

BASE_URL = "http://localhost:8000"


def test_register():
    """测试用户注册"""
    print("=" * 50)
    print("测试注册功能")
    print("=" * 50)

    url = f"{BASE_URL}/register"
    data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "test123"
    }

    try:
        response = requests.post(url, json=data)
        print(f"状态码：{response.status_code}")
        print(f"响应内容：{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 201:
            print("✓ 注册成功")
            return True
        else:
            print("✗ 注册失败")
            return False
    except Exception as e:
        print(f"✗ 请求错误：{e}")
        return False


def test_login():
    """测试用户登录"""
    print("\n" + "=" * 50)
    print("测试登录功能")
    print("=" * 50)

    url = f"{BASE_URL}/login"
    data = {
        "username": "testuser",
        "password": "test123"
    }

    try:
        response = requests.post(url, data=data)
        print(f"状态码：{response.status_code}")
        result = response.json()
        print(f"响应内容：{json.dumps(result, indent=2, ensure_ascii=False)}")

        if response.status_code == 200 and "access_token" in result:
            print("✓ 登录成功")
            print(f"Token: {result['access_token']}")
            return result['access_token']
        else:
            print("✗ 登录失败")
            return None
    except Exception as e:
        print(f"✗ 请求错误：{e}")
        return None


def test_get_current_user(token):
    """测试获取当前用户信息"""
    print("\n" + "=" * 50)
    print("测试获取用户信息")
    print("=" * 50)

    url = f"{BASE_URL}/users/me"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        print(f"状态码：{response.status_code}")
        print(f"响应内容：{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        if response.status_code == 200:
            print("✓ 获取用户信息成功")
            return True
        else:
            print("✗ 获取用户信息失败")
            return False
    except Exception as e:
        print(f"✗ 请求错误：{e}")
        return False


if __name__ == "__main__":
    print("\n开始测试 FastAPI 认证系统\n")

    # 测试注册
    register_success = test_register()

    # 测试登录
    token = test_login()

    # 如果登录成功，测试获取用户信息
    if token:
        test_get_current_user(token)

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
