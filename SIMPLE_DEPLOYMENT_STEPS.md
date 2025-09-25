# EdPrep AI - Simple Deployment Steps (FIXED)

## ✅ **Corrected Steps for Render Deployment**

### **Step 1: Deploy Backend (API)**
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. **Connect GitHub**:
   - Repository: `shanalishah/edprep-ai-prototype`
   - Branch: `main`
   - Root Directory: **Leave empty** (or use `edprep-ai-prototype`)

4. **Configure**:
   - Name: `edprep-ai-backend`
   - Environment: `Python 3`
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

5. **Environment Variables**:
   - `DEPLOYMENT_MODE` = `true`
   - `PYTHON_VERSION` = `3.11.0`

6. **Deploy**: Click "Create Web Service"
7. **Wait**: 5-10 minutes
8. **Copy the URL**: You'll get something like `https://edprep-ai-backend.onrender.com`

### **Step 2: Deploy Frontend (Website)**
1. Go back to Render dashboard
2. Click "New +" → "Static Site"
3. **Connect GitHub**:
   - Repository: `shanalishah/edprep-ai-prototype`
   - Branch: `main`
   - Root Directory: `edprep-ai-prototype` ⚠️ **Important!**

4. **Configure**:
   - Name: `edprep-ai-frontend`
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/out`

5. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = `https://edprep-ai-backend.onrender.com` (use your actual backend URL)

6. **Deploy**: Click "Create Static Site"
7. **Wait**: 3-5 minutes

## 🎯 **Key Fixes**

The issue was with the directory structure. Here's what's correct:

- **Repository Structure**: Your `frontend` folder is directly in the root
- **Root Directory**: Use `edprep-ai-prototype` (not `edprep-ai-prototype/frontend`)
- **Build Command**: Include `cd frontend` to navigate to the frontend directory
- **Publish Directory**: Use `frontend/out` (relative to the root directory)

## 🎉 **Expected Results**

- **Backend**: `https://edprep-ai-backend.onrender.com`
- **Frontend**: `https://edprep-ai-frontend.onrender.com`

## 🔧 **If You Still Get Errors**

1. **Delete the failed frontend service** in Render
2. **Create a new Static Site** with the corrected settings above
3. **Make sure Root Directory is**: `edprep-ai-prototype`
4. **Make sure Build Command is**: `cd frontend && npm install && npm run build`
5. **Make sure Publish Directory is**: `frontend/out`

The key fix is using the correct Root Directory and including `cd frontend` in the build command! 🔧
