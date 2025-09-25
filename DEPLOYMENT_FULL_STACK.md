# EdPrep AI - Full Stack Deployment Guide

## 🚀 **Complete Project Deployment on Render**

This guide will help you deploy the complete EdPrep AI project with all features:
- ✅ **Writing Test** (AI-powered essay scoring)
- ✅ **Reading Test** (Cambridge IELTS passages)
- ✅ **Listening Test** (Audio-based questions)
- ✅ **Modern UI** (Next.js frontend)
- ✅ **Advanced Backend** (FastAPI with ML models)

## 📋 **Prerequisites**

1. **GitHub Repository**: Your code is already on GitHub ✅
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Environment Variables**: Set up API keys if needed

## 🔧 **Deployment Steps**

### **Step 1: Deploy Backend (FastAPI)**

1. **Go to Render Dashboard**
   - Visit [render.com/dashboard](https://render.com/dashboard)
   - Click "New +" → "Web Service"

2. **Connect GitHub Repository**
   - Repository: `shanalishah/edprep-ai-prototype`
   - Branch: `main`
   - Root Directory: `edprep-ai-prototype`

3. **Configure Backend Service**
   - **Name**: `edprep-ai-backend`
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     cd backend && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```

4. **Environment Variables**
   - `DEPLOYMENT_MODE`: `true`
   - `PYTHON_VERSION`: `3.11.0`
   - `OPENAI_API_KEY`: (optional, for enhanced feedback)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

### **Step 2: Deploy Frontend (Next.js)**

1. **Create New Web Service**
   - Click "New +" → "Static Site"

2. **Configure Frontend Service**
   - **Name**: `edprep-ai-frontend`
   - **Repository**: `shanalishah/edprep-ai-prototype`
   - **Branch**: `main`
   - **Root Directory**: `edprep-ai-prototype/frontend`

3. **Build Settings**
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `out`

4. **Environment Variables**
   - `NEXT_PUBLIC_API_URL`: `https://edprep-ai-backend.onrender.com`

5. **Deploy**
   - Click "Create Static Site"
   - Wait for deployment (3-5 minutes)

## 🎯 **What You'll Get**

### **Backend API** (https://edprep-ai-backend.onrender.com)
- ✅ **Writing Assessment**: `/assess` endpoint
- ✅ **Reading Tests**: `/reading-tests` endpoints
- ✅ **Listening Tests**: `/listening-tests` endpoints
- ✅ **Health Check**: `/health` endpoint
- ✅ **API Documentation**: `/docs` (Swagger UI)

### **Frontend Application** (https://edprep-ai-frontend.onrender.com)
- ✅ **Modern UI**: Next.js with Tailwind CSS
- ✅ **Writing Test**: Essay submission and scoring
- ✅ **Reading Test**: Interactive reading passages
- ✅ **Listening Test**: Audio-based questions
- ✅ **Responsive Design**: Works on all devices

## 🔧 **Configuration Files**

### **Backend Requirements** (`backend/requirements.txt`)
```
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### **Frontend Configuration** (`frontend/next.config.js`)
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig
```

## 🚀 **Deployment Commands**

### **Local Testing**
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### **Production URLs**
- **Backend**: `https://edprep-ai-backend.onrender.com`
- **Frontend**: `https://edprep-ai-frontend.onrender.com`
- **API Docs**: `https://edprep-ai-backend.onrender.com/docs`

## 📊 **Features Available**

### **Writing Test**
- AI-powered essay scoring
- Four IELTS criteria evaluation
- Detailed feedback generation
- Sample prompts for practice

### **Reading Test**
- Cambridge IELTS reading passages
- Multiple question types
- Interactive question interface
- Detailed explanations

### **Listening Test**
- Audio-based questions
- Various question formats
- Comprehensive feedback
- Band score conversion

## 🔧 **Troubleshooting**

### **Backend Issues**
- Check Render logs for Python errors
- Verify environment variables
- Ensure all dependencies are in requirements.txt

### **Frontend Issues**
- Check build logs for npm errors
- Verify API URL configuration
- Ensure Next.js config is correct

### **API Connection Issues**
- Verify CORS settings in backend
- Check API URL in frontend
- Test endpoints directly in browser

## 📈 **Scaling Options**

### **Free Tier**
- Backend: 750 hours/month
- Frontend: Unlimited static hosting
- Database: 1GB PostgreSQL

### **Paid Plans**
- Backend: Always-on service
- Database: Larger storage
- Custom domains
- SSL certificates

## 🎉 **Success Indicators**

Your deployment is successful when:
- ✅ Backend API responds at `/health`
- ✅ Frontend loads without errors
- ✅ Writing test works end-to-end
- ✅ Reading test loads passages
- ✅ Listening test shows questions
- ✅ All features are accessible

## 📞 **Support**

If you encounter issues:
1. Check Render deployment logs
2. Verify GitHub repository is up to date
3. Test locally first
4. Check environment variables

---

**Your complete EdPrep AI platform will be live and accessible to users worldwide! 🌍**
