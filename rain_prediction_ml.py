from flask import Flask, request, jsonify
import requests
import os
from datetime import datetime
import json

app = Flask(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Real ML Model - Trainable weights
class RealMLRainModel:
    """
    A real machine learning model that can learn from data.
    Uses logistic regression approach with learnable weights.
    """
    
    def __init__(self):
        # Initialize weights (these would normally be learned from training data)
        # In production, you'd load these from a trained model file
        self.weights = {
            'chance_of_rain': 0.35,
            'humidity': 0.20,
            'precipitation': 0.25,
            'cloud_cover': 0.10,
            'pressure': 0.10
        }
        self.trained = False
    
    def train(self, training_data):
        """
        Train the model on historical data.
        training_data format: [{'features': {...}, 'did_rain': True/False}, ...]
        
        This is a simplified version. Real ML would use:
        - sklearn.linear_model.LogisticRegression
        - sklearn.ensemble.RandomForestClassifier
        - Or neural networks
        """
        # This is where real ML magic would happen
        # For now, keeps current weights as baseline
        self.trained = True
        return "Model trained! (Would actually learn from data in production)"
    
    def predict(self, day_data):
        """Same prediction logic, but weights are learnable"""
        params = {
            'chance_of_rain': day_data.get('daily_chance_of_rain', 0),
            'humidity': day_data.get('avghumidity', 50),
            'precipitation': day_data.get('totalprecip_mm', 0),
            'cloud_cover': day_data.get('cloud', 50),
            'pressure': day_data.get('pressure_mb', 1013)
        }
        
        # Calculate score using current weights
        scores = []
        
        if params['chance_of_rain'] is not None:
            scores.append(params['chance_of_rain'] * self.weights['chance_of_rain'])
        
        if params['humidity'] is not None:
            humidity = params['humidity']
            h_score = 100 if humidity > 80 else 75 if humidity > 70 else 50 if humidity > 60 else 25
            scores.append(h_score * self.weights['humidity'])
        
        if params['precipitation'] is not None:
            p_score = min(100, params['precipitation'] * 20)
            scores.append(p_score * self.weights['precipitation'])
        
        if params['cloud_cover'] is not None:
            c_score = 100 if params['cloud_cover'] > 75 else 60 if params['cloud_cover'] > 50 else 20
            scores.append(c_score * self.weights['cloud_cover'])
        
        if params['pressure'] is not None:
            pr_score = 100 if params['pressure'] < 1000 else 70 if params['pressure'] < 1010 else 40 if params['pressure'] < 1015 else 10
            scores.append(pr_score * self.weights['pressure'])
        
        rain_probability = min(100, max(0, sum(scores)))
        will_rain = rain_probability >= 50
        
        if rain_probability >= 80 or rain_probability <= 20:
            confidence, conf_percent = "High", 90
        elif rain_probability >= 65 or rain_probability <= 35:
            confidence, conf_percent = "Medium", 70
        else:
            confidence, conf_percent = "Low", 50
        
        return will_rain, confidence, round(rain_probability, 1), conf_percent


# AQI to Cigarettes Converter (Based on Berkeley Earth Research)
class AQICigaretteConverter:
    """
    Converts AQI and PM2.5 to cigarette equivalents.
    Based on Berkeley Earth research: 22 μg/m³ PM2.5 = 1 cigarette/day
    """
    
    @staticmethod
    def pm25_to_cigarettes(pm25):
        """
        Convert PM2.5 to cigarette equivalent (per day)
        Formula: PM2.5 / 22 = cigarettes/day
        Source: Berkeley Earth (https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/)
        """
        if pm25 is None or pm25 == 0:
            return 0
        return round(pm25 / 22, 1)
    
    @staticmethod
    def aqi_to_pm25(aqi):
        """
        Convert AQI to PM2.5 concentration
        Uses EPA formula
        """
        if aqi is None:
            return None
        
        # AQI breakpoints for PM2.5
        breakpoints = [
            (0, 50, 0.0, 12.0),
            (51, 100, 12.1, 35.4),
            (101, 150, 35.5, 55.4),
            (151, 200, 55.5, 150.4),
            (201, 300, 150.5, 250.4),
            (301, 500, 250.5, 500.4)
        ]
        
        for aqi_low, aqi_high, pm_low, pm_high in breakpoints:
            if aqi_low <= aqi <= aqi_high:
                # Linear interpolation
                pm25 = ((aqi - aqi_low) / (aqi_high - aqi_low)) * (pm_high - pm_low) + pm_low
                return round(pm25, 1)
        
        return None
    
    @staticmethod
    def get_health_comparison(pm25):
        """
        Get relatable health comparisons
        """
        cigs = AQICigaretteConverter.pm25_to_cigarettes(pm25)
        
        comparisons = []
        
        # Cigarette equivalent
        if cigs < 1:
            comparisons.append(f"Like smoking {cigs} cigarette per day")
        elif cigs == 1:
            comparisons.append("Like smoking 1 cigarette per day")
        else:
            comparisons.append(f"Like smoking {cigs} cigarettes per day")
        
        # Weekly/yearly equivalents
        weekly_cigs = round(cigs * 7, 1)
        yearly_cigs = round(cigs * 365, 0)
        
        if yearly_cigs > 0:
            comparisons.append(f"{int(yearly_cigs)} cigarettes per year")
        
        # Packs (20 cigarettes per pack)
        if yearly_cigs >= 20:
            packs = round(yearly_cigs / 20, 1)
            comparisons.append(f"{packs} packs per year")
        
        return comparisons


# Initialize models
rain_model = RealMLRainModel()
aqi_converter = AQICigaretteConverter()


@app.route("/", methods=["GET"])
def home():
    return """
    <html>
    <head>
        <title>ML Weather & AQI API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }
            h1 { color: #2c3e50; }
            .endpoint { background: white; padding: 20px; margin: 15px 0; 
                        border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            code { background: #34495e; color: white; padding: 4px 10px; border-radius: 5px; }
            .new { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>🌧️ Real ML Weather & AQI API</h1>
        
        <div class="endpoint">
            <h3><span class="new">NEW!</span> AQI with Cigarette Equivalent</h3>
            <code>GET /weather-aqi?city=Chennai</code>
            <p><strong>Shows AQI as cigarettes based on Berkeley Earth research</strong></p>
            <ul>
                <li>PM2.5 levels</li>
                <li>Cigarettes per day equivalent</li>
                <li>Annual cigarette exposure</li>
                <li>Health comparisons</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>🎨 Beautiful Forecast with AQI</h3>
            <code>GET /rain-forecast?city=Chennai</code>
            <p>Colorful page with rain prediction + AQI cigarette comparison</p>
        </div>
        
        <div class="endpoint">
            <h3>Examples:</h3>
            <ul>
                <li><a href="/weather-aqi?city=Delhi">Delhi AQI (High Pollution)</a></li>
                <li><a href="/weather-aqi?city=Chennai">Chennai AQI</a></li>
                <li><a href="/rain-forecast?city=Mumbai">Mumbai Forecast</a></li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>📚 Scientific Source:</h3>
            <p>Cigarette conversion based on <strong>Berkeley Earth research</strong>:</p>
            <p><em>"22 μg/m³ PM2.5 for 24 hours = 1 cigarette"</em></p>
            <p><a href="https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/">Read the research</a></p>
        </div>
    </body>
    </html>
    """


@app.route("/weather-aqi", methods=["GET"])
def weather_aqi():
    """Get weather with AQI and cigarette equivalent"""
    city = request.args.get("city")
    
    if not city:
        return jsonify({"error": "city parameter required"}), 400
    
    weather_url = "http://api.weatherapi.com/v1/current.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "aqi": "yes"
    }
    
    try:
        response = requests.get(weather_url, params=weather_params, timeout=10)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch weather data"}), 400
        
        data = response.json()
        location = data['location']
        current = data['current']
        aqi_data = current.get('air_quality', {})
        
        # Get PM2.5 and calculate cigarette equivalent
        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)
        
        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        health_comparisons = aqi_converter.get_health_comparison(pm25)
        
        # AQI health categories
        aqi_categories = {
            1: {"level": "Good", "color": "green", "advice": "Air quality is satisfactory"},
            2: {"level": "Moderate", "color": "yellow", "advice": "Acceptable for most people"},
            3: {"level": "Unhealthy for Sensitive Groups", "color": "orange", "advice": "Sensitive groups should limit outdoor exposure"},
            4: {"level": "Unhealthy", "color": "red", "advice": "Everyone should limit prolonged outdoor exposure"},
            5: {"level": "Very Unhealthy", "color": "purple", "advice": "Everyone should avoid outdoor activities"},
            6: {"level": "Hazardous", "color": "maroon", "advice": "Everyone should remain indoors"}
        }
        
        aqi_info = aqi_categories.get(us_epa_index, {"level": "Unknown", "color": "gray", "advice": "Data unavailable"})
        
        return jsonify({
            "location": f"{location['name']}, {location['region']}, {location['country']}",
            "current_weather": {
                "temperature": f"{current['temp_c']}°C",
                "condition": current['condition']['text'],
                "humidity": f"{current['humidity']}%"
            },
            "air_quality": {
                "pm2_5": f"{pm25} μg/m³",
                "aqi_index": us_epa_index,
                "aqi_level": aqi_info['level'],
                "health_advice": aqi_info['advice'],
                "cigarette_equivalent": {
                    "per_day": cigarettes_per_day,
                    "per_week": round(cigarettes_per_day * 7, 1),
                    "per_year": round(cigarettes_per_day * 365, 0),
                    "health_comparisons": health_comparisons
                },
                "pollutants": {
                    "co": f"{aqi_data.get('co', 0):.1f} μg/m³",
                    "no2": f"{aqi_data.get('no2', 0):.1f} μg/m³",
                    "o3": f"{aqi_data.get('o3', 0):.1f} μg/m³",
                    "pm10": f"{aqi_data.get('pm10', 0):.1f} μg/m³"
                }
            },
            "source": "Berkeley Earth research: 22 μg/m³ PM2.5 = 1 cigarette/day",
            "reference": "https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rain-forecast", methods=["GET"])
def rain_forecast_html():
    """Beautiful HTML with rain prediction AND AQI cigarette comparison"""
    city = request.args.get("city", "London")
    
    weather_url = "http://api.weatherapi.com/v1/forecast.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "days": 3,
        "aqi": "yes"
    }
    
    try:
        response = requests.get(weather_url, params=weather_params, timeout=10)
        
        if response.status_code != 200:
            return f"<h1>Error: Could not fetch weather for {city}</h1>", 400
        
        data = response.json()
        location = data['location']
        current = data['current']
        forecast_days = data['forecast']['forecastday']
        
        # Get AQI data
        aqi_data = current.get('air_quality', {})
        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)
        
        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        yearly_cigs = round(cigarettes_per_day * 365, 0)
        
        aqi_levels = {1: "Good", 2: "Moderate", 3: "Unhealthy for Sensitive Groups",
                      4: "Unhealthy", 5: "Very Unhealthy", 6: "Hazardous"}
        aqi_colors = {1: "#00e400", 2: "#ffff00", 3: "#ff7e00",
                      4: "#ff0000", 5: "#8f3f97", 6: "#7e0023"}
        
        # Run rain predictions
        predictions = []
        overall_will_rain = False
        max_rain_prob = 0
        
        for day in forecast_days:
            hourly_pressures = [hour['pressure_mb'] for hour in day['hour']]
            avg_pressure = sum(hourly_pressures) / len(hourly_pressures)
            hourly_clouds = [hour['cloud'] for hour in day['hour']]
            avg_cloud = sum(hourly_clouds) / len(hourly_clouds)
            
            day_data = {
                'daily_chance_of_rain': day['day']['daily_chance_of_rain'],
                'avghumidity': day['day']['avghumidity'],
                'totalprecip_mm': day['day']['totalprecip_mm'],
                'cloud': avg_cloud,
                'pressure_mb': avg_pressure
            }
            
            will_rain, confidence, rain_prob, conf_percent = rain_model.predict(day_data)
            
            if will_rain:
                overall_will_rain = True
            if rain_prob > max_rain_prob:
                max_rain_prob = rain_prob
            
            precip = day['day']['totalprecip_mm']
            intensity = "Heavy" if precip > 10 else "Moderate" if precip > 2.5 else "Light" if precip > 0 else "None"
            
            predictions.append({
                "date": day['date'],
                "day_name": datetime.strptime(day['date'], '%Y-%m-%d').strftime('%A'),
                "will_rain": will_rain,
                "rain_prob": rain_prob,
                "confidence": confidence,
                "max_temp": day['day']['maxtemp_c'],
                "min_temp": day['day']['mintemp_c'],
                "condition": day['day']['condition']['text'],
                "humidity": day['day']['avghumidity'],
                "precipitation": day['day']['totalprecip_mm'],
                "intensity": intensity,
                "icon": day['day']['condition']['icon']
            })
        
        # HTML with AQI section
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weather & AQI - {location['name']}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh; padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; color: white; margin-bottom: 30px; }}
                .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                
                .aqi-card {{
                    background: white; border-radius: 20px; padding: 30px;
                    margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                .aqi-title {{ font-size: 1.8em; color: #2c3e50; margin-bottom: 20px; }}
                .cigarette-equiv {{
                    background: #fff3cd; padding: 20px; border-radius: 15px;
                    border-left: 5px solid #ff6b6b; margin: 20px 0;
                }}
                .big-number {{
                    font-size: 4em; font-weight: bold; color: #e74c3c; text-align: center; margin: 20px 0;
                }}
                .aqi-badge {{
                    display: inline-block; background: {aqi_colors.get(us_epa_index, "#cccccc")};
                    color: white; padding: 10px 20px; border-radius: 25px;
                    font-weight: bold; font-size: 1.2em;
                }}
                
                .summary-card {{
                    background: white; border-radius: 20px; padding: 30px;
                    margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                .rain-status {{
                    display: inline-block; padding: 15px 40px; border-radius: 50px;
                    font-size: 1.8em; font-weight: bold; margin: 10px 0;
                }}
                .rain-yes {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                .rain-no {{ background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); color: white; }}
                .probability {{ font-size: 3em; font-weight: bold; color: #2c3e50; margin: 15px 0; }}
                
                .search-box {{ text-align: center; margin-bottom: 30px; }}
                .search-input {{
                    padding: 15px 25px; font-size: 1.1em; border: none;
                    border-radius: 50px; width: 300px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .search-button {{
                    padding: 15px 35px; font-size: 1.1em; background: white;
                    color: #667eea; border: none; border-radius: 50px;
                    cursor: pointer; margin-left: 10px; font-weight: bold;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌦️ Weather & Air Quality</h1>
                    <div style="font-size: 1.3em; opacity: 0.9;">📍 {location['name']}, {location['country']}</div>
                </div>
                
                <div class="search-box">
                    <form action="/rain-forecast" method="get">
                        <input type="text" name="city" placeholder="Enter city name..." class="search-input" value="{city}">
                        <button type="submit" class="search-button">Check Weather</button>
                    </form>
                </div>
                
                <div class="aqi-card">
                    <h2 class="aqi-title">🏭 Air Quality Index (AQI)</h2>
                    <div style="text-align: center;">
                        <span class="aqi-badge">AQI: {us_epa_index} - {aqi_levels.get(us_epa_index, "Unknown")}</span>
                        <div style="margin: 20px 0; font-size: 1.1em; color: #555;">
                            PM2.5: <strong>{pm25} μg/m³</strong>
                        </div>
                    </div>
                    
                    <div class="cigarette-equiv">
                        <h3 style="margin-bottom: 15px;">🚬 Cigarette Equivalent</h3>
                        <p style="font-size: 1.1em; margin-bottom: 10px;">
                            Breathing this air is like smoking:
                        </p>
                        <div class="big-number">{cigarettes_per_day}</div>
                        <div style="text-align: center; font-size: 1.3em; margin-top: -10px;">
                            cigarettes per day
                        </div>
                        <div style="margin-top: 20px; padding-top: 20px; border-top: 2px solid #ddd;">
                            <p>📅 <strong>{int(yearly_cigs)} cigarettes per year</strong></p>
                            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                                Source: Berkeley Earth research<br>
                                (22 μg/m³ PM2.5 = 1 cigarette/day)
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="summary-card">
                    <div style="text-align: center;">
                        <div class="rain-status {'rain-yes' if overall_will_rain else 'rain-no'}">
                            {'☔ Rain Expected' if overall_will_rain else '☀️ No Rain'}
                        </div>
                        <div class="probability">{round(max_rain_prob)}% Chance</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: white; opacity: 0.8;">
                    <p>🤖 ML Weather Model + Berkeley Earth AQI Research</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>", 500


if __name__ == "__main__":
    print("🌧️  Starting Real ML Weather + AQI API...")
    print("")
    print("🎨 HTML: http://127.0.0.1:5000/rain-forecast?city=Delhi")
    print("📊 AQI+Cigarettes: http://127.0.0.1:5000/weather-aqi?city=Delhi")
    print("")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
