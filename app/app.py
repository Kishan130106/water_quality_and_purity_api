# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from database.database_main import db, init_db, get_user_by_id, get_latest_water_data, insert_water_data
import logging
from datetime import datetime
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
CORS(app,resources={r"/api/*": {"origins": "https://v0-water-quality-monitor.vercel.app/"}})

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize a flag to track if DB is initialized
db_initialized = False

@app.route('/')
def index():
    """Show all available API endpoints"""
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':  # Skip static files
            routes.append({
                'endpoint': rule.endpoint,
                'path': str(rule),
                'methods': list(rule.methods)
            })
    return jsonify({
        'message': 'Water Quality API is running!',
        'endpoints': routes
    })

@app.before_request
def initialize_db_on_first_request():
    """Initialize database on first request"""
    global db_initialized
    if not db_initialized:
        try:
            if db.test_connection():
                init_db()
                logging.info("Database initialized successfully")
            else:
                logging.error("Failed to connect to database")
            db_initialized = True
        except Exception as e:
            logging.error(f"Database initialization error: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if db.test_connection():
            return jsonify({"status": "healthy", "database": "connected"})
        else:
            return jsonify({"status": "unhealthy", "database": "disconnected"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    try:
        with db.get_cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM public.users ORDER BY user_id ASC")
            users = cur.fetchall()
            return jsonify(users)
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    try:
        user = get_user_by_id(user_id)
        if user:
            return jsonify(user)
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logging.error(f"Error fetching user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/water-data', methods=['GET'])
def get_water_data():
    """Get water quality data"""
    try:
        device_id = request.args.get('device_id')
        limit = request.args.get('limit', 100, type=int)
        
        data = get_latest_water_data(device_id, limit)
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error fetching water data: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/iot/water-data', methods=['POST'])
def receive_iot_data():
    """Receive data from IoT devices"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['device_id', 'ph_value', 'turbidity', 'temperature', 'tds_value']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Insert data
        timestamp = insert_water_data(
            device_id=data['device_id'],
            ph_value=data['ph_value'],
            turbidity=data['turbidity'],
            temperature=data['temperature'],
            tds_value=data['tds_value'],
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            status=data.get('status', 'active')
        )
        
        if timestamp:
            return jsonify({
                "message": "Data received successfully",
                "timestamp": timestamp.isoformat() if timestamp else None
            }), 201
        else:
            return jsonify({"error": "Failed to insert data"}), 500
            
    except Exception as e:
        logging.error(f"Error processing IoT data: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic statistics about water data"""
    try:
        with db.get_cursor(cursor_factory=RealDictCursor) as cur:
            # Get total records count
            cur.execute("SELECT COUNT(*) as total_records FROM iot_water_data")
            total = cur.fetchone()
            
            # Get device count
            cur.execute("SELECT COUNT(DISTINCT device_id) as device_count FROM iot_water_data")
            devices = cur.fetchone()
            
            # Get latest reading time
            cur.execute("SELECT MAX(time) as latest_reading FROM iot_water_data")
            latest = cur.fetchone()
            
            return jsonify({
                "total_records": total['total_records'],
                "device_count": devices['device_count'],
                "latest_reading": latest['latest_reading'].isoformat() if latest['latest_reading'] else None
            })
            
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/analyze-water', methods=['GET'])
def analyze_water():
    """Analyze water quality with predictions"""
    try:
        # Lazy import to avoid circular imports
        from models.waterapi import get_water_analysis
        device_id = request.args.get('device_id')
        limit = request.args.get('limit', 10, type=int)
        
        analysis = get_water_analysis(device_id, limit)
        return jsonify(analysis)
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return jsonify({"error": "Analysis module not available"}), 500
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({"error": "Analysis failed"}), 500

@app.route('/api/predict-water', methods=['POST'])
def predict_water():
    """Predict water quality from parameters"""
    try:
        # Lazy import to avoid circular imports
        from models.waterapi import predict_water_quality
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        prediction = predict_water_quality(
            data.get('ph', 7.0),
            data.get('turbidity', 5.0),
            data.get('temperature', 25.0),
            data.get('tds', 120.0)
        )
        return jsonify(prediction)
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return jsonify({"error": "Prediction module not available"}), 500
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/api/water-quality-report', methods=['GET'])
def water_quality_report():
    """Generate comprehensive water quality report"""
    try:
        # Lazy import to avoid circular imports
        from models.waterapi import get_water_analysis
        device_id = request.args.get('device_id')
        limit = request.args.get('limit', 24, type=int)
        
        analysis = get_water_analysis(device_id, limit)
        
        # Create a timestamp for the report
        current_time = datetime.now().isoformat()
        report_id = f"water_quality_{device_id or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build the report safely
        report = {
            "report_id": report_id,
            "generated_at": current_time,
            "device_id": device_id,
            "analysis_period": f"Last {limit} readings"
        }
        
        # Add analysis data if it's a dictionary
        if isinstance(analysis, dict):
            report.update({
                "recent_readings": analysis.get("recent_readings", []),
                "statistics": analysis.get("statistics", {}),
                "trends": analysis.get("trends", {}),
                "recommendations": analysis.get("recommendations", [])
            })
        else:
            # Handle case where analysis returned an error
            report["error"] = "Analysis failed"
        
        return jsonify(report)
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return jsonify({"error": "Analysis module not available"}), 500
    except Exception as e:
        logging.error(f"Report generation error: {e}")
        return jsonify({"error": "Report generation failed"}), 500

if __name__ == '__main__':
    # Initialize database before running the app
    try:
        if db.test_connection():
            init_db()
            logging.info("Database initialized successfully")
        else:
            logging.error("Failed to connect to database")
    except Exception as e:
        logging.error(f"Startup error: {e}")
    
    # Show all routes when starting
    with app.app_context():
        print("Available routes:")
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static':
                methods = ','.join(rule.methods)
                print(f"  {rule.endpoint:30} {methods:20} {rule}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)