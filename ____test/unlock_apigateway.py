import requests, datetime, hashlib, time
seconds = 2
def get_accesstoken(client_id='87ed6cf1e9274e65af6500193fd7dce8', 
                    clientsecret='5e56225a865fc7368f7e1e57b5bdd0fc', 
                    username='trinhnhattuyen12a4@gmail.com', 
                    password='nhattuyen0414'):
    print("Get access token ...")
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    url = "https://euapi.sciener.com/oauth2/token"
    
    data = {
        "clientId": client_id,
        "clientSecret": clientsecret,
        "username": username,
        "password": hashlib.md5(password.encode()).hexdigest()
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print(response.json())
        return response.json()['access_token'], response.json()['refresh_token']
    else:
        return "Failed to get access token"

def refresh_accesstoken(refresh_token,
                        client_id='87ed6cf1e9274e65af6500193fd7dce8', 
                        clientsecret='5e56225a865fc7368f7e1e57b5bdd0fc', 
                        grant_type='refresh_token'):
    print("Refresh access token ...")
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    url = "https://euapi.sciener.com/oauth2/token"
    
    data = {
        "clientId": client_id,
        "clientSecret": clientsecret,
        "grant_type": grant_type,
        "refresh_token": refresh_token
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print(response.json())
        return response.json()['access_token'], response.json()['refresh_token']
    else:
        return "Failed to get access token"
       
def remote_lock(access_token, lock_id='9399008', client_id='87ed6cf1e9274e65af6500193fd7dce8', lock=False):
    # access_token = get_accesstoken()
    url = "https://euapi.sciener.com/v3/lock/unlock"
    if lock:
        url = "https://euapi.sciener.com/v3/lock/lock"
    
    now = datetime.datetime.now()
    new_time = now + datetime.timedelta(seconds=seconds)
    timestamp = int(new_time.timestamp() * 1000)
    
    data = {
        "clientId": client_id,
        "accessToken": access_token,
        "lockId": lock_id,
        "date": timestamp
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        lock_info = response.json()
        print(lock_info)
        if lock:
            print('Lock Successful!')
        else:
            print('Unlock Successful!')
    else:
        if lock:
            print('Lock Fail!')
        else:
            print('Unlock Fail!')
        print(f"Error: {response.status_code} - {response.text}")

def detail(access_token, lock_id='9399008', client_id='87ed6cf1e9274e65af6500193fd7dce8'):
    # access_token = get_accesstoken()
    
    
    now = datetime.datetime.now()
    new_time = now + datetime.timedelta(seconds=seconds)
    date = int(new_time.timestamp() * 1000)

    url = f"https://euapi.sciener.com/v3/lock/detail?clientId={client_id}&accessToken={access_token}&lockId={lock_id}&date={date}"
    response = requests.post(url)
    if response.status_code == 200:
        lock_info = response.json()
        print(lock_info)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
try:
    accesstoken, refreshtoken = get_accesstoken()
    # accesstoken, refreshtoken = refresh_accesstoken(refresh_token=refreshtoken)
    remote_lock(accesstoken)
    time.sleep(2)
    remote_lock(accesstoken, lock=True)
    # detail(accesstoken)
except:
    print("Unlock Fail !")