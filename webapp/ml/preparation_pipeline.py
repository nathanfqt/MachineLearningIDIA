import pandas as pd 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

def preparation(data):
    
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    else:
        data = data.copy()
    
    #Timestamp
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%Y-%m-%d %H:%M:%S")
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour + (data['Timestamp'].dt.minute/60) + (data['Timestamp'].dt.second/3600)
    data = data.drop(columns=['Timestamp'])
    
    
    #IP Addresses
    import geoip2.database
    
    city_reader = geoip2.database.Reader(str(ASSETS_DIR / "GeoLite2-City.mmdb"))
    asn_reader = geoip2.database.Reader(str(ASSETS_DIR / "GeoLite2-ASN.mmdb"))
    
    def get_ip_details(ip):
        
        asn_num, asn_org, city, country, lat, long = None, None, None, None, None, None
    
        try:
            city_res = city_reader.city(ip)
            asn_res = asn_reader.asn(ip)
            
            asn_num = asn_res.autonomous_system_number
            asn_org = asn_res.autonomous_system_organization
            city = city_res.city.name
            country = city_res.country.name
            lat = city_res.location.latitude
            long = city_res.location.longitude
    
        except Exception as e:
            pass 
    
        return pd.Series([asn_num, asn_org, city, country, lat, long])
    
    data[['asn_num Source', 'asn_org Source', 'city Source', 'country Source', 'lat Source', 'long Source']] = data['Source IP Address'].apply(get_ip_details)
    data[['asn_num Destination', 'asn_org Destination', 'city Destination', 'country Destination', 'lat Destination', 'long Destination']] = data['Destination IP Address'].apply(get_ip_details)
    
    def classify_asn(org_name):
        org_name = str(org_name).lower()
        
        if any(k in org_name for k in ['google', 'amazon', 'aws', 'microsoft', 'azure', 'oracle', 'ibm', 'alibaba', # Giants
                                        'ovh', 'hetzner', 'digitalocean', 'linode', 'vultr', 'contabo', 'scaleway', 'clouvider',  # Hosts
                                        'akamai', 'cloudflare', 'fastly', 'leaseweb', 'equinix', 'rackspace',         # CDN
                                        'hosting', 'cloud', 'datacenter', 'vps', 'server', 'compute', 'softlayer'    # General
                                      ]  ):
            return 'Cloud/Hosting'
    
        if any(k in org_name for k in ['telecom', 'telekom', 'telecoms', 'provider', 'broadband', 'communication', 'communications',
        'mobile', 'wireless', 'network', 'online', 'internet',
        'orange', 'proxad', 'free', 'sfr', 'numericable', 'cegetel', 'bouygues', 
        'comcast', 'verizon', 'at&t', 't-mobile', 'vodafone', 'charter', 'spectrum',
        'cox', 'centurytel', 'frontier', 'bt-central', 'telefonica', 'deutsche telekom',
        'chinanet', 'unicom', 'reliance', 'jio', 'airtel', 'ntt', 'kpn', 'bredband2', 'te data', 'videotron']):
           
            return 'Internet Provider'
        
        if any(k in org_name for k in ['academy', 'university', 'college', 'research', 'education', 'state', 'department', 'institute', 'ministry', 'ministere', 'administration', 
        'council',  'defense', 'military', 'national', 'regione', 'headquarters', 'district',
        'city of', 'uninet']):
            return 'Education/Gov'
        
        if any(k in org_name for k in ['holding', 'bank', 'corp', 'inc', 'ltd', 'llc', 'company', 'co.', 'corporation', 's.a', 'sa', 'as', 'a.s', 'a.d', 'ad', 'ag', 'a.g', 'gmbh', 'ltda', 's.a.u'
                                      'insurance', 'industries', 'logistics', 'srl', 's.r.l']):
            return 'Enterprise'
            
        return 'Enterprise'
    
    data['asn_source type'] = data['asn_org Source'].apply(classify_asn)
    data['asn_dest type'] = data['asn_org Destination'].apply(classify_asn)
    
    data.drop(['asn_org Destination', 'asn_org Source', 'city Destination', 'city Source'], axis=1, inplace=True)
    
    import ipaddress
    
    def get_info_from_ip(ip_string):
        ip = ipaddress.ip_address(ip_string)
        return pd.Series([ip.version, ip.is_private, ip.is_global, ip.is_link_local, ip.is_loopback, ip.is_multicast, ip.is_reserved, ip.is_unspecified, int(ip)])
    
    
    data[["Destination IP Version", "Destination IP Private", "Destination IP Global", "Destination IP Link Local", "Destination IP loopback", "Destination IP Multicast", "Destination IP Reserved", "Destination IP Unspecified", "Destination IP Int"]] = data["Destination IP Address"].apply(get_info_from_ip)
    data[["Source IP Version", "Source IP Private", "Source IP Global", "Source IP Link Local", "Source IP loopback", "Source IP Multicast", "Source IP Reserved", "Source IP Unspecified", "Source IP Int"]] = data["Source IP Address"].apply(get_info_from_ip)
    
    col_dest = ["Destination IP Version", "Destination IP Global", "Destination IP Link Local", "Destination IP loopback", "Destination IP Multicast", "Destination IP Reserved", "Destination IP Unspecified"]
    col_src = ["Source IP Version", "Source IP Global", "Source IP Link Local", "Source IP loopback", "Source IP Multicast", "Source IP Reserved", "Source IP Unspecified"]
    data.drop(col_dest, axis=1, inplace=True)
    data.drop(col_src, axis=1, inplace=True)
    
    data[['octet1 Source', 'octet2 Source', 'octet3 Source', 'octet4 Source']] = data['Source IP Address'].str.split('.', expand=True).astype(int)
    data[['octet1 Destination', 'octet2 Destination', 'octet3 Destination', 'octet4 Destination']] = data['Destination IP Address'].str.split('.', expand=True).astype(int)
    
    data.drop(columns=['Source IP Address', 'Destination IP Address'], axis=1, inplace=True)
    
    import numpy as np
    
    def haversine_np(lat1, lon1, lat2, lon2):
        R = 6371.0 
    
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi / 2.0)**2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    data['dist Source/Dest'] = haversine_np(
        data['lat Source'], data['long Source'], 
        data['lat Destination'], data['long Destination']
    )
    
    #Port
    def port_category(port):
        if 0 <= port <= 1023:
            return "Well Known"
        elif 1024 <= port <= 49151:
            return "Registered"
        elif 49152 <= port <= 65535:
            return "Ephemeral"
    
    data["Categorical Source Port"] = data["Source Port"].apply(port_category)
    data["Categorical Destination Port"] = data["Destination Port"].apply(port_category)
    
    
    #Payload
    data["len payload"] = data["Payload Data"].str.len()
    
    import numpy as np
    from collections import Counter
    
    def entropy(string):
        
        if len(string) == 0:
            return 0
        
        counts = Counter(string)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
    
    
        return -(probs * np.log2(probs)).sum()
        
    
    data["entropy payload"] = data["Payload Data"].apply(entropy)
    
    data.drop(columns=['Payload Data'], axis=1, inplace=True)
    
    data['Malware Indicators'] = data['Malware Indicators'].notna().astype(int)
    data['Alerts/Warnings'] = data['Alerts/Warnings'].notna().astype(int)
    
    data.drop(columns=['User Information'], axis=1, inplace=True)
    
    
    #Device Information
    from ua_parser import parse
    
    def parse_device_info(ua_string):
        ua = parse(ua_string)
        browser = ua.user_agent.family if ua.user_agent is not None else None
        browser_version = ua.user_agent.major if ua.user_agent is not None else None
        os = ua.os.family if ua.os is not None else None
        os_version = ua.os.major if ua.os is not None else None
        device = ua.device.model if ua.device is not None else "Other"
    
        s = ua_string.split(")")[1]
        engine = s.split("/")[0]
        engine = engine.replace(" ","")
        if engine == "":
            engine = "Trident"
    
        return browser, browser_version, os, os_version, device, engine
    
    data[["Browser", "Browser Version", "OS", "OS Version", "Device", "Engine"]] = data['Device Information'].apply(lambda x: pd.Series(parse_device_info(x)))

    data.drop(columns=['Device Information'], axis=1, inplace=True)
    
    data['Firewall Logs'] = data['Firewall Logs'].notna().astype(int)
    data['IDS/IPS Alerts'] = data['IDS/IPS Alerts'].notna().astype(int)
    
    
    #Geo location
    data["city"] = data["Geo-location Data"].str.split(",", expand=True)[0]
    data['state'] = data["Geo-location Data"].str.split(",", expand=True)[1]
    data.drop(columns=['Geo-location Data'], axis=1, inplace=True)
    data2 = pd.read_csv(ASSETS_DIR / "worldcities.csv")
    data2 = data2[['city_ascii', 'lat', 'lng', 'population', 'country']]
    data2 = data2[data2['country']=='India']
    data2.drop(columns='country', inplace = True)
    data2.columns = ['city', 'lat', 'long', 'pop']
    data2 = data2.round(2)
    
    data3 = pd.read_csv(ASSETS_DIR / "final_cities.csv")
    data3 = data3[['City', 'Latitude', 'Longitude', 'Population']]
    data3.columns = ['city', 'lat', 'long', 'pop']
    data3 = data3.round(2)
    
    columns = ["name", "population", "location"]
    
    fichiers = [
        ASSETS_DIR / "place_city.ndjson",
        ASSETS_DIR / "place-hamlet.ndjson",
        ASSETS_DIR / "place-village.ndjson",
        ASSETS_DIR / "place-town.ndjson"
    ]
    
    datas = [pd.read_json(f, lines=True) for f in fichiers]
    
    data4 = pd.concat(datas, ignore_index=True)
    data4 = data4[columns]
    data4[["lon", "lat"]] = data4["location"].apply(pd.Series)
    data4 = data4[['name', 'lat', 'lon', 'population', 'location']]
    data4.drop(columns='location', inplace = True)
    data4.columns = ['city', 'lat', 'long', 'pop']
    data4 = data4.round(2)
    
    data_cities = pd.concat([data2, data3, data4], ignore_index=True)
    data_cities = data_cities.drop_duplicates()
    
    d2 = data2.drop_duplicates(subset=['city']).set_index('city')
    d3 = data3.drop_duplicates(subset=['city']).set_index('city')
    d4 = data4.drop_duplicates(subset=['city']).set_index('city')
    
    def find_infos(city):
    
        cols = ['lat', 'long', 'pop']
    
        if city in d4.index:
            return d4.loc[city]
        
        elif city in d2.index:
            return d2.loc[city]
        
        elif city in d3.index:
            return d3.loc[city]
    
        else:
            return pd.Series({c: None for c in cols})
    
    data[['lat', 'long', 'pop']] = data['city'].apply(find_infos)
    
    
    
    #Proxy

    data[['asn_num Proxy', 'asn_org Proxy', 'city Proxy', 'country Proxy', 'lat Proxy', 'long Proxy']] = data['Proxy Information'].apply(get_ip_details)
    data['Proxy Information'] = data['Proxy Information'].fillna(0)
    
    def get_int_proxy(ip_string):
        ip = ipaddress.ip_address(ip_string)
        return pd.Series([int(ip)])
    
    data['Proxy Information'] = data['Proxy Information'].apply(get_int_proxy)  
    data['asn_proxy type'] = data['asn_org Proxy'].apply(classify_asn)
    data.drop(columns=['asn_org Proxy', 'city Proxy'], axis=1, inplace=True)
    
    #data.to_csv('extracted data.csv', index=False)
    
    data = pd.get_dummies(data, columns=['Protocol', 
                                         'Packet Type',
                                         'Traffic Type',
                                         'Log Source',
                                        'Categorical Source Port',
                                        'Categorical Destination Port'],  drop_first=True)
    
    
    data.drop(columns=['Attack Signature', 'Action Taken', 'Severity Level', 'Network Segment'], inplace=True)
    data['Proxy Information'] = data['Proxy Information'].notnull().astype(int)
    
    data.dropna(subset=['asn_num Destination'], inplace=True)
    data.dropna(subset=['asn_num Source'], inplace=True)
    data.dropna(subset=['country Source', 'country Destination'], inplace=True)
    
    data = pd.get_dummies(data, columns=['asn_source type', 'asn_dest type'],  drop_first=True)
    
    def modify_browser(browser):
        if "Safari" in browser:
            return "Safari"
        if "Chrome" in browser:
            return "Chrome"
        if "Firefox" in browser:
            return "Firefox"
        return browser
    
    data['Browser'] = data['Browser'].apply(modify_browser)
    
    data = pd.get_dummies(data, columns=['Browser'],  drop_first=True)
    
    
    data.dropna(subset=['OS Version'], inplace=True)

    data = pd.get_dummies(data, columns=['OS'],  drop_first=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    data['OS Version'] = le.fit_transform(data['OS Version'])
    
    def modify_device(device):
        if device in ["iPhone", "iPod", "iPad", "UG", "Smartphone", "Tablet"]:
            return "Mobile"
        else:
            return "Desktop"
    
    data['Device'] = data['Device'].apply(modify_device)
    data = pd.get_dummies(data, columns=['Device', 'Engine'],  drop_first=True)
    
    le = LabelEncoder()
    
    data['city'] = le.fit_transform(data['city'])
    
    le = LabelEncoder()
    
    data['state'] = le.fit_transform(data['state'])
    
    data.dropna(subset=['lat'], inplace=True)
    
    import numpy as np
    data['pop'] = data['pop'].fillna(0)
    
    data.drop(columns=['asn_num Proxy', 'country Proxy', 'lat Proxy', 'long Proxy', 'asn_proxy type'], inplace = True)
    
    from sklearn.preprocessing import LabelEncoder
    
    all_countries = pd.concat([data['country Source'], data['country Destination']]).unique()
    
    le_countries = LabelEncoder()
    le_countries.fit(all_countries)
    
    data['country Source'] = le_countries.transform(data['country Source'])
    data['country Destination'] = le_countries.transform(data['country Destination'])
    
    data.drop(columns=['state'], inplace = True)                      
    data.drop(columns=['octet4 Source',
    'octet1 Destination',                         
    'octet2 Destination',                         
    'octet3 Destination',                         
    'octet4 Destination'], inplace=True)
    data.drop(columns=['country Source', 'country Destination'], inplace=True)
    
    return data



    
    
    
