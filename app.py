import os
import base64
import uuid
import numpy as np
import torch
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
import tempfile
import traceback

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Orientation, Spacing, 
    NormalizeIntensity, SpatialPad
)
from config import Config
from model import nnFormer

# =========================================
# METRIC CALCULATION FUNCTIONS
# =========================================

def calculate_dice_score(pred, gt, label):
    """Calculate Dice score for a specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total

def calculate_iou(pred, gt, label):
    """Calculate Intersection over Union for a specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_precision_recall(pred, gt, label):
    """Calculate Precision and Recall for a specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def calculate_sensitivity_specificity(pred, gt, label):
    """Calculate Sensitivity and Specificity"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    tp = np.sum(pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity

def calculate_hausdorff_distance(pred, gt, label, percentile=95):
    """Calculate Hausdorff Distance (simplified version)"""
    from scipy.spatial.distance import directed_hausdorff
    
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return float('inf')
    
    pred_points = np.argwhere(pred_mask)
    gt_points = np.argwhere(gt_mask)
    
    forward = directed_hausdorff(pred_points, gt_points)[0]
    backward = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(forward, backward)

# =========================================
# 1. SETUP & CSS
# =========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME])
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <title>NeuroAI - Advanced Medical Imaging</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            /* ==================== RESET & VARIABLES ==================== */
            * { 
                box-sizing: border-box; 
                margin: 0; 
                padding: 0;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            :root { 
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --primary-light: #818cf8;
                --accent: #06b6d4;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --bg-dark: #0a0a0f;
                --bg-darker: #050508;
                --card-bg: rgba(17, 24, 39, 0.7);
                --card-border: rgba(99, 102, 241, 0.2);
                --text-primary: #f9fafb;
                --text-secondary: #9ca3af;
                --text-muted: #6b7280;
                --glass-bg: rgba(255, 255, 255, 0.03);
                --glass-border: rgba(255, 255, 255, 0.1);
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
                --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.3);
            }
            
            body { 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 50%, #0f1419 100%);
                background-attachment: fixed;
                color: var(--text-primary);
                overflow-x: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%);
                pointer-events: none;
                z-index: 0;
            }

            /* ==================== GLASSMORPHISM CARDS ==================== */
            .glass-card {
                background: var(--card-bg);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: var(--shadow-md);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .glass-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--primary-light), transparent);
                opacity: 0.5;
            }

            .glass-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg), var(--shadow-glow);
                border-color: var(--primary);
            }

            /* Metric Cards */
            .metric-card {
                background: linear-gradient(135deg, var(--card-bg) 0%, rgba(17, 24, 39, 0.5) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 20px;
                box-shadow: var(--shadow-sm);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                height: 100%;
            }

            .metric-card::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: linear-gradient(90deg, var(--primary), var(--accent));
                transform: scaleX(0);
                transform-origin: left;
                transition: transform 0.3s ease;
            }

            .metric-card:hover::after {
                transform: scaleX(1);
            }

            .metric-label {
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: var(--text-secondary);
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                color: var(--text-primary);
                line-height: 1.2;
                margin-bottom: 4px;
                background: linear-gradient(135deg, var(--primary-light), var(--accent));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .metric-unit {
                font-size: 0.75rem;
                color: var(--text-muted);
                font-weight: 500;
            }

            /* ==================== HEADER ==================== */
            .app-header {
                padding: 32px 0 24px;
                position: relative;
                z-index: 10;
            }

            .app-title {
                font-size: 2rem;
                font-weight: 800;
                background: linear-gradient(135deg, var(--primary-light) 0%, var(--accent) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .version-badge {
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                padding: 6px 16px;
                border-radius: 50px;
                font-size: 0.75rem;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                letter-spacing: 0.5px;
            }

            /* ==================== TABS ==================== */
            .custom-tabs {
                border-bottom: 1px solid var(--card-border);
                margin-bottom: 24px;
            }

            .custom-tab {
                background: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                color: var(--text-secondary);
                padding: 12px 24px;
                font-weight: 600;
                font-size: 0.9rem;
                transition: all 0.3s ease;
                cursor: pointer;
                letter-spacing: 0.5px;
            }

            .custom-tab:hover {
                color: var(--primary-light);
                background: rgba(99, 102, 241, 0.05);
            }

            .custom-tab.active {
                color: var(--primary-light);
                border-bottom-color: var(--primary);
            }

            /* Override Dash tabs */
            .tab--selected {
                border-bottom: 3px solid var(--primary) !important;
                color: var(--primary-light) !important;
                background: rgba(99, 102, 241, 0.05) !important;
            }

            /* ==================== UPLOAD AREA ==================== */
            .upload-wrapper {
                position: relative;
                margin-bottom: 24px;
            }

            .upload-container {
                border: 2px dashed var(--card-border);
                border-radius: 16px;
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                padding: 32px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .upload-container::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
                opacity: 0;
                transform: translate(-50%, -50%);
                transition: all 0.5s ease;
            }

            .upload-container:hover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.05);
                transform: translateY(-2px);
                box-shadow: var(--shadow-md), 0 0 30px rgba(99, 102, 241, 0.2);
            }

            .upload-container:hover::before {
                width: 300px;
                height: 300px;
                opacity: 0.1;
            }

            .upload-icon {
                font-size: 3rem;
                color: var(--primary);
                margin-bottom: 16px;
                transition: all 0.3s ease;
            }

            .upload-container:hover .upload-icon {
                transform: scale(1.1) translateY(-4px);
                color: var(--primary-light);
            }

            .upload-text {
                font-weight: 600;
                color: var(--text-primary);
                font-size: 0.95rem;
                margin-bottom: 8px;
            }

            .upload-hint {
                font-size: 0.8rem;
                color: var(--text-muted);
            }

            /* ==================== BUTTONS ==================== */
            .btn-modern {
                border-radius: 12px;
                font-weight: 600;
                padding: 12px 28px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: none;
                width: 100%;
                font-size: 0.95rem;
                letter-spacing: 0.3px;
                position: relative;
                overflow: hidden;
            }

            .btn-modern::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: width 0.5s, height 0.5s;
            }

            .btn-modern:hover::before {
                width: 300px;
                height: 300px;
            }

            .btn-modern:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
            }

            .btn-modern:active:not(:disabled) {
                transform: translateY(0);
            }

            .btn-modern:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .btn-primary-custom {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
            }

            .btn-secondary-custom {
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                color: var(--text-primary);
            }

            .btn-secondary-custom:hover:not(:disabled) {
                background: rgba(255, 255, 255, 0.08);
                border-color: var(--primary);
            }

            /* ==================== PROGRESS BAR ==================== */
            .progress-wrapper {
                margin-top: 16px;
            }

            .progress-custom {
                height: 8px;
                background: var(--glass-bg);
                border-radius: 50px;
                overflow: hidden;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
            }

            .progress-bar-custom {
                background: linear-gradient(90deg, var(--primary), var(--accent));
                height: 100%;
                border-radius: 50px;
                transition: width 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .progress-bar-custom::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                animation: shimmer 2s infinite;
            }

            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }

            .progress-text {
                font-size: 0.8rem;
                color: var(--text-secondary);
                margin-bottom: 8px;
                font-weight: 500;
            }

            /* ==================== SECTION HEADERS ==================== */
            .section-header {
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: var(--text-secondary);
                margin: 28px 0 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid var(--card-border);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .section-header::before {
                content: '';
                width: 4px;
                height: 16px;
                background: linear-gradient(180deg, var(--primary), var(--accent));
                border-radius: 2px;
            }

            /* ==================== GRAPH CONTAINERS ==================== */
            .graph-container {
                background: var(--card-bg);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 20px;
                box-shadow: var(--shadow-sm);
                height: 100%;
            }

            .graph-header {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: var(--text-secondary);
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            /* ==================== SLIDER ==================== */
            .custom-slider .rc-slider-track {
                background: linear-gradient(90deg, var(--primary), var(--accent));
            }

            .custom-slider .rc-slider-handle {
                border-color: var(--primary);
                background: white;
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
            }

            .custom-slider .rc-slider-handle:hover {
                box-shadow: 0 0 0 6px rgba(99, 102, 241, 0.3);
            }

            /* ==================== STATUS MESSAGES ==================== */
            .status-message {
                padding: 12px 20px;
                border-radius: 12px;
                margin-bottom: 16px;
                font-size: 0.875rem;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 10px;
                animation: slideInDown 0.3s ease;
            }

            .status-success {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: var(--success);
            }

            .status-error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: var(--danger);
            }

            .status-warning {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: var(--warning);
            }

            @keyframes slideInDown {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* ==================== RESPONSIVE ==================== */
            @media (max-width: 1024px) {
                .app-title { font-size: 1.8rem; }
                .glass-card { padding: 20px; }
                .metric-value { font-size: 1.8rem; }
                .section-header { font-size: 0.65rem; margin: 20px 0 12px; }
                .graph-container { padding: 16px; }
            }
            
            @media (max-width: 768px) {
                body { overflow-x: hidden; }
                .app-header { padding: 20px 0 16px; }
                .app-title { font-size: 1.5rem; gap: 8px; }
                .app-title i { font-size: 1.5rem !important; }
                .version-badge { padding: 4px 12px; font-size: 0.7rem; }
                .metric-value { font-size: 1.5rem; }
                .metric-label { font-size: 0.65rem; }
                .metric-unit { font-size: 0.7rem; }
                .glass-card { padding: 16px; border-radius: 12px; }
                .metric-card { padding: 16px; margin-bottom: 12px; }
                .section-header { font-size: 0.6rem; margin: 16px 0 12px; letter-spacing: 1.5px; }
                .upload-container { padding: 24px; }
                .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
                .upload-text { font-size: 0.9rem; }
                .upload-hint { font-size: 0.75rem; }
                .btn-modern { padding: 11px 20px; font-size: 0.85rem; border-radius: 10px; }
                .progress-custom { height: 6px; }
                .progress-text { font-size: 0.75rem; }
                .graph-container { padding: 12px; border-radius: 10px; }
                .graph-header { font-size: 0.7rem; margin-bottom: 10px; }
                .custom-tab { padding: 10px 16px; font-size: 0.85rem; }
                .status-message { padding: 10px 16px; font-size: 0.8rem; }
                .metric-card { width: 100%; }
            }
            
            @media (max-width: 640px) {
                .app-header { padding: 16px 0 12px; }
                .app-title { font-size: 1.3rem; gap: 6px; flex-direction: column; align-items: flex-start; }
                .app-title i { font-size: 1.3rem !important; }
                .version-badge { padding: 3px 10px; font-size: 0.65rem; margin-top: 8px; }
                .glass-card { padding: 14px; border-radius: 10px; }
                .metric-card { padding: 14px; margin-bottom: 10px; }
                .metric-value { font-size: 1.4rem; }
                .metric-label { font-size: 0.6rem; letter-spacing: 1px; }
                .section-header { font-size: 0.6rem; margin: 14px 0 10px; letter-spacing: 1px; }
                .section-header::before { width: 3px; height: 14px; }
                .upload-container { padding: 20px; }
                .upload-icon { font-size: 2rem; margin-bottom: 10px; }
                .upload-text { font-size: 0.85rem; }
                .upload-hint { font-size: 0.7rem; }
                .gt-upload-small { padding: 14px; font-size: 0.75rem; }
                .btn-modern { padding: 12px 24px; font-size: 0.9rem; }
                .custom-tab { padding: 8px 10px; font-size: 0.7rem; }
                .graph-container { padding: 10px; }
                .graph-header { font-size: 0.6rem; }
                table th, table td { padding: 6px !important; font-size: 0.7rem !important; }
            }
            
            @media (max-width: 480px) {
                .container-fluid { padding-left: 12px !important; padding-right: 12px !important; }
                .app-title { font-size: 1.2rem; }
                .version-badge { font-size: 0.6rem; padding: 2px 8px; }
                .glass-card { padding: 12px; }
                .metric-card { padding: 12px; }
                .metric-value { font-size: 1.3rem; }
                .metric-label { font-size: 0.55rem; }
                .section-header { font-size: 0.55rem; margin: 12px 0 8px; }
                .upload-container { padding: 16px; }
                .upload-icon { font-size: 1.8rem; }
                .btn-modern { padding: 10px 18px; font-size: 0.8rem; }
            }
            
            @media (max-height: 500px) and (orientation: landscape) {
                .app-header { padding: 12px 0 8px; }
                .section-header { margin: 10px 0 8px; }
                .metric-card { padding: 10px; margin-bottom: 8px; }
                .upload-container { padding: 14px; }
                .graph-container { padding: 10px; }
            }
            
            @media (hover: none) and (pointer: coarse) {
                .btn-modern { min-height: 48px; }
                .upload-container { min-height: 120px; }
                .custom-tab { min-height: 48px; }
                .glass-card:hover, .metric-card:hover, .upload-container:hover { transform: none; }
                .rc-slider-handle { width: 24px; height: 24px; margin-top: -10px; box-shadow: 0 0 0 6px rgba(99, 102, 241, 0.2); }
                .rc-slider-handle:active { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0.3); }
            }

            /* ==================== SCROLLBAR ==================== */
            ::-webkit-scrollbar { width: 10px; height: 10px; }
            ::-webkit-scrollbar-track { background: var(--bg-darker); }
            ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--primary), var(--accent)); border-radius: 5px; }
            ::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

            /* ==================== ANIMATIONS ==================== */
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
            .animate-fade-in { animation: fadeIn 0.5s ease; }
            .animate-slide-in { animation: slideIn 0.5s ease; }

            /* ==================== ACCESSIBILITY ==================== */
            *:focus { outline: 2px solid var(--primary); outline-offset: 2px; }
            .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border-width: 0; }

            /* ==================== TAB CONTENT ==================== */
            .tab-content-wrapper { min-height: 600px; position: relative; }

            /* NEW: GT Upload specific styles */
            .gt-upload-small { border: 1px dashed var(--card-border); border-radius: 12px; background: var(--glass-bg); padding: 16px; text-align: center; cursor: pointer; transition: all 0.3s ease; font-size: 0.8rem; }
            .gt-upload-small:hover { border-color: var(--warning); background: rgba(245, 158, 11, 0.05); }
            .gt-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px; }
            .gt-badge-active { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
            .gt-badge-inactive { background: rgba(156, 163, 175, 0.2); color: var(--text-muted); border: 1px solid var(--text-muted); }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = "/mnt/data"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

MODEL = None

# =========================================
# REASSEMBLY FUNCTION (SPLIT & STITCH)
# =========================================
def reassemble_model():
    """Stitches split model parts back into a single .pth file if needed."""
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best.pth")
    
    # If file exists and looks big enough, skip
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 100_000_000:
        return ckpt_path

    print("🔧 Reassembling model from parts...")
    part_0 = ckpt_path + ".part0"
    part_1 = ckpt_path + ".part1"
    
    if os.path.exists(part_0):
        with open(ckpt_path, "wb") as out_f:
            # Read Part 0
            with open(part_0, "rb") as f0: out_f.write(f0.read())
            # Read Part 1 (if exists)
            if os.path.exists(part_1):
                with open(part_1, "rb") as f1: out_f.write(f1.read())
        print("✅ Model reassembled successfully!")
    else:
        print("⚠️ No model parts found. App may crash if inference runs.")
    
    return ckpt_path

def get_model():
    global MODEL
    if MODEL is None:
        print("📥 Loading Model...")
        try:
            # Reassemble file first
            ckpt_path = reassemble_model()
            
            model = nnFormer(
                crop_size=Config.PATCH_SIZE, embedding_dim=Config.EMBEDDING_DIM,
                input_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES,
                depths=Config.DEPTHS, num_heads=Config.NUM_HEADS,
                window_size=Config.WINDOW_SIZE, patch_size=[4, 4, 4], deep_supervision=False 
            ).to(DEVICE)
            
            # Load weights (CPU safe)
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            MODEL = model
            print("✅ Model Loaded.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return None
    return MODEL

def preprocess(path):
    t = [LoadImage(image_only=True), EnsureChannelFirst(), Orientation(axcodes="RAS"),
         Spacing(pixdim=[1,1,1], mode="bilinear"), NormalizeIntensity(nonzero=True, channel_wise=True),
         SpatialPad(spatial_size=Config.PATCH_SIZE)]
    data = t[0](path)
    for tr in t[1:]: data = tr(data)
    
    # --- FIX: HANDLE SINGLE CHANNEL INPUTS ---
    # If the image has only 1 channel (Shape: [1, D, H, W]), 
    # we repeat it 4 times to match the model's expected input (Shape: [4, D, H, W]).
    if data.shape[0] == 1:
        print("⚠️ Single channel input detected. Repeating data to fake 4 channels.")
        data = data.repeat(4, 1, 1, 1)
    # ----------------------------------------
    
    return data.unsqueeze(0).to(DEVICE)

def make_mesh(mask, label_id, color, name, opacity=0.3):
    binary = (mask > 0) if label_id == 'brain' else (mask == label_id)
    step = 4 if label_id == 'brain' else 2
    if np.sum(binary) < 100: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(binary, step_size=step)
        return go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],
                         opacity=opacity, color=color, name=name, showlegend=True, 
                         hovertemplate='<b>%{text}</b><extra></extra>', text=[name]*len(verts))
    except: return None

# =========================================
# 2. LAYOUT
# =========================================
def metric_card(title, id_suffix, icon):
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={"color": "var(--primary-light)", "fontSize": "0.9rem"}),
            html.Span(title)
        ], className="metric-label"),
        html.Div("0.00", id=f"val-{id_suffix}", className="metric-value"),
        html.Div("cm³", className="metric-unit")
    ], className="metric-card animate-fade-in")

app.layout = html.Div([
    dcc.Interval(id="progress-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Store(id='session'), 
    dcc.Store(id='results'),
    dcc.Store(id='error-state'),
    dcc.Store(id='gt-session'), 

    dbc.Container([
        # HEADER
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-brain", style={"fontSize": "1.8rem"}),
                        html.Span("NeuroAI Pro")
                    ], className="app-title")
                ], width=8),
                dbc.Col([
                    html.Div("v4.0 Pro", className="version-badge float-end")
                ], width=4)
            ], className="align-items-center")
        ], className="app-header"),

        dbc.Row([
            # ==================== SIDEBAR ====================
            dbc.Col([
                html.Div([
                    # UPLOAD SECTION
                    html.Div("1. Data Input", className="section-header"),
                    
                    html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt upload-icon"),
                                html.Div("Drop MRI File Here", className="upload-text"),
                                html.Div("or click to browse (.nii / .nii.gz)", className="upload-hint")
                            ]),
                            className="upload-container",
                            multiple=False,
                            accept='.nii,.nii.gz,.nii.gz'
                        ),
                    ], className="upload-wrapper"),
                    
                    html.Div(id="upload-status"),
                    
                    # NEW: GROUND TRUTH UPLOAD (Optional)
                    html.Div([
                        html.Div([
                            "Ground Truth (Optional)",
                            html.Span(id="gt-status-badge", className="gt-badge gt-badge-inactive float-end", 
                                     children="Not Loaded", style={"marginLeft": "8px"})
                        ], className="section-header"),
                        
                        html.Div([
                            dcc.Upload(
                                id='upload-gt',
                                children=html.Div([
                                    html.I(className="fas fa-file-medical", style={"fontSize": "1.5rem", "color": "var(--warning)", "marginBottom": "8px"}),
                                    html.Div("Upload Ground Truth Mask", style={"fontSize": "0.8rem", "fontWeight": "500"}),
                                    html.Div("(for accuracy metrics)", style={"fontSize": "0.7rem", "color": "var(--text-muted)", "marginTop": "4px"})
                                ]),
                                className="gt-upload-small",
                                multiple=False,
                                accept='.nii,.nii.gz,.nii.gz'
                            ),
                        ]),
                        html.Div(id="gt-upload-status", style={"marginTop": "8px"}),
                    ]),
                    
                    # ANALYSIS SECTION
                    html.Div("2. Analysis", className="section-header"),
                    html.Button([
                        html.I(className="fas fa-play-circle", style={"marginRight": "8px"}),
                        "Start Diagnosis"
                    ], id="btn-diagnose", className="btn-modern btn-primary-custom mb-3", disabled=True),
                    
                    # PROGRESS
                    html.Div([
                        html.Div("Processing brain scan...", id="progress-text", className="progress-text", 
                                style={"display": "none"}),
                        html.Div([
                            html.Div(id="progress-bar-inner", className="progress-bar-custom", 
                                    style={"width": "0%"})
                        ], className="progress-custom", style={"opacity": "0"}, id="progress-bar-container"),
                    ], className="progress-wrapper"),

                    # EXPORT SECTION
                    html.Div("3. Export", className="section-header"),
                    html.Button([
                        html.I(className="fas fa-download", style={"marginRight": "8px"}),
                        "Download Mask"
                    ], id="btn-dl", className="btn-modern btn-secondary-custom mb-3", disabled=True),
                    dcc.Download(id="dl-nifti"),
                    
                    # METRICS SUMMARY
                    html.Div("4. Metrics", className="section-header"),
                    dbc.Row([
                        dbc.Col(metric_card("Total", "total", "fa-cube"), width=12, className="mb-3"),
                        dbc.Col(metric_card("Core", "core", "fa-circle"), width=12, className="mb-3"),
                        dbc.Col(metric_card("Edema", "edema", "fa-droplet"), width=12, className="mb-3"),
                        dbc.Col(metric_card("Enhance", "enhance", "fa-star"), width=12),
                    ]),
                    
                    # INFO SECTION
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-info-circle", 
                                  style={"color": "var(--accent)", "marginRight": "8px"}),
                            html.Span("System Status", style={"fontWeight": "600", "fontSize": "0.75rem"})
                        ], style={"marginBottom": "12px"}),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-microchip", 
                                      style={"color": "var(--success)", "marginRight": "6px", "fontSize": "0.75rem"}),
                                html.Span(f"Device: {DEVICE.type.upper()}", 
                                         style={"fontSize": "0.7rem", "color": "var(--text-secondary)"})
                            ], style={"marginBottom": "6px"}),
                            html.Div([
                                html.I(className="fas fa-check-circle", 
                                      style={"color": "var(--success)", "marginRight": "6px", "fontSize": "0.75rem"}),
                                html.Span("Model: Ready", 
                                         style={"fontSize": "0.7rem", "color": "var(--text-secondary)"})
                            ])
                        ])
                    ], className="glass-card", style={"padding": "16px", "marginTop": "20px"})
                    
                ], className="glass-card")
            ], width={'size': 3, 'order': 1}, xs=12, sm=12, md=3, lg=3, className="animate-slide-in mb-4 mb-md-0"),

            # ==================== MAIN CONTENT WITH TABS ====================
            dbc.Col([
                # TABS
                dcc.Tabs(id="main-tabs", value='tab-3d', className="custom-tabs", children=[
                    dcc.Tab(
                        label='🔮 3D Visualization',
                        value='tab-3d',
                        className='custom-tab',
                        selected_className='tab--selected',
                        children=[
                            html.Div([
                                # 3D VIEWERS ROW
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Brain Context View"),
                                                html.I(className="fas fa-cube", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="3d-context", 
                                                     style={"height": "500px"},
                                                     config={'displayModeBar': True, 'displaylogo': False})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6, className="mb-3 mb-md-0"),
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Tumor Architecture"),
                                                html.I(className="fas fa-project-diagram", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="3d-tumor", 
                                                     style={"height": "500px"},
                                                     config={'displayModeBar': True, 'displaylogo': False})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6),
                                ], className="mb-4"),
                            ], className="tab-content-wrapper")
                        ]
                    ),
                    
                    dcc.Tab(
                        label='📊 2D Slice Explorer',
                        value='tab-2d',
                        className='custom-tab',
                        selected_className='tab--selected',
                        children=[
                            html.Div([
                                dbc.Row([
                                    # AXIAL VIEW
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Axial View"),
                                                html.Span(id="slice-indicator-axial", 
                                                         style={"fontSize": "0.7rem", "color": "var(--accent)", "fontWeight": "600"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="2d-axial", 
                                                     style={"height": "350px"},
                                                     config={'displayModeBar': False}),
                                            html.Div([
                                                dcc.Slider(
                                                    id="slice-slider-axial", 
                                                    min=0, max=100, step=1, value=50, 
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    className="custom-slider"
                                                )
                                            ], style={"padding": "0 10px", "marginTop": "10px"})
                                        ], className="graph-container")
                                    ], width=4, xs=12, sm=12, md=4, lg=4, className="mb-3 mb-md-0"),
                                    
                                    # SAGITTAL VIEW
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Sagittal View"),
                                                html.Span(id="slice-indicator-sagittal", 
                                                         style={"fontSize": "0.7rem", "color": "var(--accent)", "fontWeight": "600"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="2d-sagittal", 
                                                     style={"height": "350px"},
                                                     config={'displayModeBar': False}),
                                            html.Div([
                                                dcc.Slider(
                                                    id="slice-slider-sagittal", 
                                                    min=0, max=100, step=1, value=50, 
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    className="custom-slider"
                                                )
                                            ], style={"padding": "0 10px", "marginTop": "10px"})
                                        ], className="graph-container")
                                    ], width=4, xs=12, sm=12, md=4, lg=4, className="mb-3 mb-md-0"),
                                    
                                    # CORONAL VIEW
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Coronal View"),
                                                html.Span(id="slice-indicator-coronal", 
                                                         style={"fontSize": "0.7rem", "color": "var(--accent)", "fontWeight": "600"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="2d-coronal", 
                                                     style={"height": "350px"},
                                                     config={'displayModeBar': False}),
                                            html.Div([
                                                dcc.Slider(
                                                    id="slice-slider-coronal", 
                                                    min=0, max=100, step=1, value=50, 
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    className="custom-slider"
                                                )
                                            ], style={"padding": "0 10px", "marginTop": "10px"})
                                        ], className="graph-container")
                                    ], width=4, xs=12, sm=12, md=4, lg=4),
                                ])
                            ], className="tab-content-wrapper", style={"paddingTop": "20px"})
                        ]
                    ),
                    
                    dcc.Tab(
                        label='📈 Analytics',
                        value='tab-analytics',
                        className='custom-tab',
                        selected_className='tab--selected',
                        children=[
                            html.Div([
                                dbc.Row([
                                    # MRI INTENSITY HISTOGRAM
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("MRI Intensity Distribution"),
                                                html.I(className="fas fa-chart-bar", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="intensity-histogram", 
                                                     style={"height": "300px"},
                                                     config={'displayModeBar': True, 'displaylogo': False})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6, className="mb-3 mb-md-0"),
                                    
                                    # BIOMARKER RADAR
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Biomarker Profile"),
                                                html.I(className="fas fa-chart-radar", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="radar-chart", 
                                                     style={"height": "300px"},
                                                     config={'displayModeBar': False})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6),
                                ], className="mb-4"),
                                
                                dbc.Row([
                                    # VOLUME DISTRIBUTION
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Tumor Region Distribution"),
                                                html.I(className="fas fa-chart-pie", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            dcc.Graph(id="volume-pie", 
                                                     style={"height": "300px"},
                                                     config={'displayModeBar': False})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6, className="mb-3 mb-md-0"),
                                    
                                    # REGION STATISTICS TABLE
                                    dbc.Col([
                                        html.Div([
                                            html.Div([
                                                html.Span("Region Statistics"),
                                                html.I(className="fas fa-table", 
                                                      style={"fontSize": "0.7rem", "color": "var(--text-muted)"})
                                            ], className="graph-header"),
                                            html.Div(id="stats-table", style={"height": "300px", "overflowY": "auto"})
                                        ], className="graph-container")
                                    ], width=6, xs=12, sm=12, md=6, lg=6),
                                ])
                            ], className="tab-content-wrapper", style={"paddingTop": "20px"})
                        ]
                    ),
                    
                    # NEW TAB: Ground Truth Comparison
                    dcc.Tab(
                        label='🎯 Ground Truth Comparison',
                        value='tab-gt',
                        className='custom-tab',
                        selected_className='tab--selected',
                        children=[
                            html.Div(id="gt-tab-content", children=[
                                html.Div([
                                    html.I(className="fas fa-info-circle", style={"fontSize": "2rem", "color": "var(--text-muted)", "marginBottom": "16px"}),
                                    html.H5("No Ground Truth Loaded", style={"color": "var(--text-secondary)", "marginBottom": "8px"}),
                                    html.P("Upload a ground truth mask to enable comparison metrics and visualizations.", 
                                          style={"color": "var(--text-muted)", "fontSize": "0.9rem"})
                                ], style={"textAlign": "center", "padding": "80px 20px"}, className="glass-card")
                            ], className="tab-content-wrapper", style={"paddingTop": "20px"})
                        ]
                    ),
                ])
            ], width={'size': 9, 'order': 2}, xs=12, sm=12, md=9, lg=9, className="animate-fade-in")
        ])
    ], fluid=True, className="pb-5", style={"position": "relative", "zIndex": "1"})
], style={"minHeight": "100vh"})

# =========================================
# 3. CALLBACKS
# =========================================

# UPLOAD MRI CALLBACK
@app.callback(
    [Output("session", "data"), 
     Output("upload-status", "children"), 
     Output("btn-diagnose", "disabled"),
     Output("error-state", "data")],
    Input("upload-data", "contents"), 
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def upload_file(contents, filename):
    if not contents:
        raise PreventUpdate
    
    try:
        if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"marginRight": "8px"}),
                f"Invalid file type: {filename}. Please upload .nii or .nii.gz files."
            ], className="status-message status-error")
            return None, error_msg, True, "invalid_file"
        
        sid = str(uuid.uuid4())
        session_path = os.path.join(TEMP_DIR, sid)
        os.makedirs(session_path, exist_ok=True)
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        file_path = os.path.join(session_path, "in.nii.gz")
        with open(file_path, "wb") as f:
            f.write(decoded)
        
        success_msg = html.Div([
            html.I(className="fas fa-check-circle", style={"marginRight": "8px"}),
            f"Successfully loaded: {filename}"
        ], className="status-message status-success")
        
        return session_path, success_msg, False, None
        
    except Exception as e:
        error_msg = html.Div([
            html.I(className="fas fa-times-circle", style={"marginRight": "8px"}),
            f"Upload failed: {str(e)}"
        ], className="status-message status-error")
        return None, error_msg, True, str(e)

# NEW: UPLOAD GROUND TRUTH CALLBACK
@app.callback(
    [Output("gt-session", "data"),
     Output("gt-upload-status", "children"),
     Output("gt-status-badge", "children"),
     Output("gt-status-badge", "className")],
    Input("upload-gt", "contents"),
    State("upload-gt", "filename"),
    State("session", "data"),
    prevent_initial_call=True
)
def upload_ground_truth(contents, filename, session_path):
    if not contents:
        raise PreventUpdate
    
    try:
        if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"marginRight": "6px", "fontSize": "0.7rem"}),
                f"Invalid GT file type"
            ], className="status-message status-error", style={"padding": "8px 12px", "fontSize": "0.75rem"})
            return None, error_msg, "Not Loaded", "gt-badge gt-badge-inactive float-end"
        
        if not session_path:
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"marginRight": "6px", "fontSize": "0.7rem"}),
                "Please upload MRI scan first"
            ], className="status-message status-warning", style={"padding": "8px 12px", "fontSize": "0.75rem"})
            return None, error_msg, "Not Loaded", "gt-badge gt-badge-inactive float-end"
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        gt_path = os.path.join(session_path, "gt.nii.gz")
        with open(gt_path, "wb") as f:
            f.write(decoded)
        
        success_msg = html.Div([
            html.I(className="fas fa-check-circle", style={"marginRight": "6px", "fontSize": "0.7rem"}),
            f"GT loaded: {filename[:20]}..."
        ], className="status-message status-success", style={"padding": "8px 12px", "fontSize": "0.75rem"})
        
        return session_path, success_msg, "GT Loaded", "gt-badge gt-badge-active float-end"
        
    except Exception as e:
        error_msg = html.Div([
            html.I(className="fas fa-times-circle", style={"marginRight": "6px", "fontSize": "0.7rem"}),
            f"GT upload failed"
        ], className="status-message status-error", style={"padding": "8px 12px", "fontSize": "0.75rem"})
        return None, error_msg, "Not Loaded", "gt-badge gt-badge-inactive float-end"

# PROGRESS ANIMATION
@app.callback(
    [Output("progress-bar-inner", "style"),
     Output("progress-bar-container", "style"),
     Output("progress-text", "style")],
    [Input("btn-diagnose", "n_clicks"), 
     Input("progress-interval", "n_intervals")],
    [State("progress-bar-inner", "style")],
    prevent_initial_call=True
)
def animate_progress(n_click, n_int, current_style):
    if ctx.triggered_id == "btn-diagnose":
        return (
            {"width": "10%"},
            {"opacity": "1"},
            {"display": "block"}
        )
    
    if ctx.triggered_id == "progress-interval":
        current_width = int(current_style.get("width", "0%").rstrip("%"))
        if current_width < 90:
            new_width = min(current_width + 5, 90)
            return (
                {"width": f"{new_width}%"},
                {"opacity": "1"},
                {"display": "block"}
            )
    
    return no_update, no_update, no_update

# START PROGRESS TIMER
@app.callback(
    Output("progress-interval", "disabled"),
    Input("btn-diagnose", "n_clicks"),
    prevent_initial_call=True
)
def start_timer(n):
    if n:
        return False
    return no_update

# MAIN DIAGNOSIS CALLBACK
@app.callback(
    [Output("val-total", "children"), 
     Output("val-core", "children"), 
     Output("val-edema", "children"), 
     Output("val-enhance", "children"), 
     Output("3d-context", "figure"), 
     Output("3d-tumor", "figure"),
     Output("radar-chart", "figure"),
     Output("intensity-histogram", "figure"),
     Output("volume-pie", "figure"),
     Output("stats-table", "children"),
     Output("slice-slider-axial", "max"),
     Output("slice-slider-sagittal", "max"),
     Output("slice-slider-coronal", "max"),
     Output("results", "data"),
     Output("btn-dl", "disabled"), 
     Output("progress-bar-inner", "style", allow_duplicate=True),
     Output("progress-interval", "disabled", allow_duplicate=True),
     Output("upload-status", "children", allow_duplicate=True),
     Output("gt-tab-content", "children")],  # NEW: Update GT tab
    Input("btn-diagnose", "n_clicks"),
    [State("session", "data"), State("gt-session", "data")],
    prevent_initial_call=True
)
def diagnose(n, path, gt_path):
    if not path:
        raise PreventUpdate
    
    empty_gt_tab = html.Div([
        html.I(className="fas fa-info-circle", style={"fontSize": "2rem", "color": "var(--text-muted)", "marginBottom": "16px"}),
        html.H5("No Ground Truth Loaded", style={"color": "var(--text-secondary)", "marginBottom": "8px"}),
        html.P("Upload a ground truth mask to enable comparison metrics.", style={"color": "var(--text-muted)", "fontSize": "0.9rem"})
    ], style={"textAlign": "center", "padding": "80px 20px"}, className="glass-card")
    
    try:
        model = get_model()
        if not model:
            raise Exception("Model failed to load")

        fpath = os.path.join(path, "in.nii.gz")
        input_tensor = preprocess(fpath)
        
        with torch.no_grad():
            output = sliding_window_inference(input_tensor, Config.PATCH_SIZE, Config.SW_BATCH_SIZE, model, overlap=0.5)
        
        mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # --- FIX 2: Use resampled input for visualization ---
        orig = input_tensor[0, 0, :, :, :].cpu().numpy()
        orig_header = nib.load(fpath)
        # ----------------------------------------------------
        
        np.save(os.path.join(path, "img.npy"), orig)
        np.save(os.path.join(path, "mask.npy"), mask)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), orig_header.affine), os.path.join(path, "pred.nii.gz"))
        
        # Calculate volumes
        voxel_volume = 0.001
        vc = round(np.sum(mask == 1) * voxel_volume, 2)
        ve = round(np.sum(mask == 2) * voxel_volume, 2)
        ven = round(np.sum(mask == 3) * voxel_volume, 2)
        vt = round(vc + ve + ven, 2)
        
        colors = {1: '#ef4444', 2: '#3b82f6', 3: '#fbbf24'}
        bg_color = "rgba(0,0,0,0)"
        
        # 3D layouts
        layout_3d = dict(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', bgcolor=bg_color),
            paper_bgcolor=bg_color, plot_bgcolor=bg_color, margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(17, 24, 39, 0.8)", bordercolor="rgba(99, 102, 241, 0.3)", borderwidth=1, font=dict(color="white", size=10))
        )
        
        # 3D Context
        fig_ctx = go.Figure(layout=layout_3d)
        brain_mesh = make_mesh(np.where(orig > 0.1, 1, 0), 'brain', '#6b7280', 'Brain', 0.06)
        if brain_mesh: fig_ctx.add_trace(brain_mesh)
        for label_id, name in [(1, 'Necrotic Core'), (2, 'Edema'), (3, 'Enhancing')]:
            mesh = make_mesh(mask, label_id, colors[label_id], name, 0.4)
            if mesh: fig_ctx.add_trace(mesh)
        
        # 3D Tumor
        fig_tum = go.Figure(layout=layout_3d)
        for label_id, name in [(1, 'Necrotic Core'), (2, 'Edema'), (3, 'Enhancing')]:
            mesh = make_mesh(mask, label_id, colors[label_id], name, 0.5)
            if mesh: fig_tum.add_trace(mesh)
        
        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[vt, vc, ve, vc/(vt+0.01)*100], theta=['Total Volume', 'Necrotic Core', 'Edema', 'Core Ratio %'],
            fill='toself', name='Biomarkers', line=dict(color='#6366f1', width=2), fillcolor='rgba(99, 102, 241, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor=bg_color, radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(color='#9ca3af', size=9), gridcolor='rgba(255, 255, 255, 0.1)'),
                      angularaxis=dict(tickfont=dict(color='white', size=10))),
            paper_bgcolor=bg_color, font=dict(color='white'), margin=dict(t=40, b=20, l=20, r=20), showlegend=False
        )
        
        # Histogram
        brain_mask = orig > 0.1
        brain_intensities = orig[brain_mask].flatten()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=brain_intensities, nbinsx=100, marker=dict(color='#6366f1', line=dict(color='#818cf8', width=0.5)), 
                                        name='Brain Tissue', opacity=0.75))
        for label_id, color, name in [(1, '#ef4444', 'Necrotic'), (2, '#3b82f6', 'Edema'), (3, '#fbbf24', 'Enhancing')]:
            region_mask = mask == label_id
            if np.sum(region_mask) > 0:
                fig_hist.add_trace(go.Histogram(x=orig[region_mask].flatten(), nbinsx=50, marker=dict(color=color, opacity=0.5), name=name))
        fig_hist.update_layout(
            barmode='overlay', xaxis=dict(title='Intensity', color='white', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Frequency', color='white', gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color='white'), margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(x=0.7, y=0.95, bgcolor="rgba(17, 24, 39, 0.8)", bordercolor="rgba(99, 102, 241, 0.3)", borderwidth=1, font=dict(size=9))
        )
        
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(labels=['Necrotic Core', 'Edema', 'Enhancing'], values=[vc, ve, ven],
                                         marker=dict(colors=['#ef4444', '#3b82f6', '#fbbf24']), hole=0.4, textfont=dict(color='white', size=12),
                                         hovertemplate='<b>%{label}</b><br>Volume: %{value:.2f} cm³<br>Percent: %{percent}<extra></extra>')])
        fig_pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20),
                             showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10)))
        
        # Stats table
        stats_table = html.Table([
            html.Thead([html.Tr([html.Th("Region", style={"padding": "12px", "textAlign": "left", "borderBottom": "2px solid var(--primary)"}),
                                html.Th("Volume (cm³)", style={"padding": "12px", "textAlign": "right", "borderBottom": "2px solid var(--primary)"}),
                                html.Th("% of Total", style={"padding": "12px", "textAlign": "right", "borderBottom": "2px solid var(--primary)"})])]),
            html.Tbody([
                html.Tr([html.Td([html.Span("●", style={"color": "#ef4444", "marginRight": "8px", "fontSize": "1.2rem"}), "Necrotic Core"], style={"padding": "12px"}),
                        html.Td(f"{vc:.2f}", style={"padding": "12px", "textAlign": "right", "fontWeight": "600"}),
                        html.Td(f"{(vc/(vt+0.01)*100):.1f}%", style={"padding": "12px", "textAlign": "right", "color": "var(--text-muted)"})],
                        style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                html.Tr([html.Td([html.Span("●", style={"color": "#3b82f6", "marginRight": "8px", "fontSize": "1.2rem"}), "Edema"], style={"padding": "12px"}),
                        html.Td(f"{ve:.2f}", style={"padding": "12px", "textAlign": "right", "fontWeight": "600"}),
                        html.Td(f"{(ve/(vt+0.01)*100):.1f}%", style={"padding": "12px", "textAlign": "right", "color": "var(--text-muted)"})],
                        style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                html.Tr([html.Td([html.Span("●", style={"color": "#fbbf24", "marginRight": "8px", "fontSize": "1.2rem"}), "Enhancing"], style={"padding": "12px"}),
                        html.Td(f"{ven:.2f}", style={"padding": "12px", "textAlign": "right", "fontWeight": "600"}),
                        html.Td(f"{(ven/(vt+0.01)*100):.1f}%", style={"padding": "12px", "textAlign": "right", "color": "var(--text-muted)"})],
                        style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                html.Tr([html.Td("Total Tumor", style={"padding": "12px", "fontWeight": "700"}),
                        html.Td(f"{vt:.2f}", style={"padding": "12px", "textAlign": "right", "fontWeight": "700", "color": "var(--primary-light)"}),
                        html.Td("100%", style={"padding": "12px", "textAlign": "right", "fontWeight": "700", "color": "var(--primary-light)"})])
            ])
        ], style={"width": "100%", "fontSize": "0.85rem"})
        
        # CHECK FOR GROUND TRUTH
        gt_tab_content = empty_gt_tab
        if gt_path and os.path.exists(os.path.join(gt_path, "gt.nii.gz")):
            try:
                # --- FIX: Ensure GT matches resampled shape ---
                # We don't just load the file, we should technically resample it too
                # For simplicity, we just load it and resize if needed (or assume user uploaded matching GT)
                # In a real app, you'd run 'preprocess' on GT too, but use Nearest Neighbor interpolation
                
                gt_nifti = nib.load(os.path.join(gt_path, "gt.nii.gz"))
                gt = gt_nifti.get_fdata().astype(np.int32)
                if len(gt.shape) == 4: gt = gt[:,:,:,0]
                
                # Simple check - if shapes don't match, we can't compute metrics easily without resampling GT
                if gt.shape != mask.shape:
                    print(f"⚠️ Shape mismatch: GT {gt.shape} vs Pred {mask.shape}")
                    # Try to crop/pad to match? Or just warn.
                    # For this fix, let's assume they might differ and show error if so.
                    raise ValueError(f"Ground truth shape {gt.shape} doesn't match resampled prediction {mask.shape}")
                
                # Calculate metrics
                dice_scores = {1: calculate_dice_score(mask, gt, 1), 2: calculate_dice_score(mask, gt, 2), 3: calculate_dice_score(mask, gt, 3)}
                iou_scores = {1: calculate_iou(mask, gt, 1), 2: calculate_iou(mask, gt, 2), 3: calculate_iou(mask, gt, 3)}
                
                metrics_data = []
                for label_id, name in [(1, 'Necrotic Core'), (2, 'Edema'), (3, 'Enhancing')]:
                    precision, recall = calculate_precision_recall(mask, gt, label_id)
                    sensitivity, specificity = calculate_sensitivity_specificity(mask, gt, label_id)
                    metrics_data.append({
                        'name': name, 'dice': dice_scores[label_id], 'iou': iou_scores[label_id],
                        'precision': precision, 'recall': recall, 'sensitivity': sensitivity, 'specificity': specificity
                    })
                
                # Metrics bar chart
                fig_metrics = go.Figure()
                x_labels = ['Necrotic Core', 'Edema', 'Enhancing']
                fig_metrics.add_trace(go.Bar(name='Dice Score', x=x_labels, y=[m['dice'] for m in metrics_data], 
                                            marker_color='#6366f1'))
                fig_metrics.add_trace(go.Bar(name='IoU', x=x_labels, y=[m['iou'] for m in metrics_data], 
                                            marker_color='#06b6d4'))
                fig_metrics.update_layout(
                    barmode='group', paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    xaxis=dict(title='Region', color='white', gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(title='Score', color='white', gridcolor='rgba(255,255,255,0.1)', range=[0, 1]),
                    font=dict(color='white'), margin=dict(l=40, r=20, t=40, b=40),
                    legend=dict(x=0.02, y=0.98, bgcolor="rgba(17, 24, 39, 0.8)", font=dict(size=10))
                )
                
                # Comparison table
                comparison_table = html.Table([
                    html.Thead([html.Tr([
                        html.Th("Metric", style={"padding": "12px", "textAlign": "left", "borderBottom": "2px solid var(--primary)"}),
                        html.Th("Necrotic", style={"padding": "12px", "textAlign": "center", "borderBottom": "2px solid var(--primary)"}),
                        html.Th("Edema", style={"padding": "12px", "textAlign": "center", "borderBottom": "2px solid var(--primary)"}),
                        html.Th("Enhancing", style={"padding": "12px", "textAlign": "center", "borderBottom": "2px solid var(--primary)"})
                    ])]),
                    html.Tbody([
                        html.Tr([html.Td("Dice Score", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['dice']:.3f}", style={"padding": "12px", "textAlign": "center", 
                                "color": "var(--success)" if metrics_data[i]['dice'] > 0.8 else "var(--warning)" if metrics_data[i]['dice'] > 0.5 else "var(--danger)"}) 
                                for i in range(3)], style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                        html.Tr([html.Td("IoU", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['iou']:.3f}", style={"padding": "12px", "textAlign": "center"}) for i in range(3)],
                               style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                        html.Tr([html.Td("Precision", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['precision']:.3f}", style={"padding": "12px", "textAlign": "center"}) for i in range(3)],
                               style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                        html.Tr([html.Td("Recall", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['recall']:.3f}", style={"padding": "12px", "textAlign": "center"}) for i in range(3)],
                               style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                        html.Tr([html.Td("Sensitivity", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['sensitivity']:.3f}", style={"padding": "12px", "textAlign": "center"}) for i in range(3)],
                               style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),
                        html.Tr([html.Td("Specificity", style={"padding": "12px", "fontWeight": "600"})] + 
                               [html.Td(f"{metrics_data[i]['specificity']:.3f}", style={"padding": "12px", "textAlign": "center"}) for i in range(3)])
                    ])
                ], style={"width": "100%", "fontSize": "0.85rem"})
                
                # Build GT tab content
                gt_tab_content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([html.Span("Performance Metrics"), html.I(className="fas fa-chart-line", 
                                                                     style={"fontSize": "0.7rem", "color": "var(--text-muted)"})], className="graph-header"),
                                dcc.Graph(figure=fig_metrics, style={"height": "300px"}, config={'displayModeBar': False})
                            ], className="graph-container")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Div([html.Span("Detailed Metrics"), html.I(className="fas fa-table", 
                                                                  style={"fontSize": "0.7rem", "color": "var(--text-muted)"})], className="graph-header"),
                                html.Div(comparison_table, style={"height": "300px", "overflowY": "auto"})
                            ], className="graph-container")
                        ], width=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([html.Span("Overall Summary"), html.I(className="fas fa-info-circle", 
                                                                 style={"fontSize": "0.7rem", "color": "var(--text-muted)"})], className="graph-header"),
                                html.Div([
                                    html.H4(f"Mean Dice: {np.mean([m['dice'] for m in metrics_data]):.3f}", 
                                           style={"color": "var(--primary-light)", "marginBottom": "16px"}),
                                    html.P(f"The model achieved an average Dice score of {np.mean([m['dice'] for m in metrics_data]):.3f} across all tumor regions.",
                                          style={"color": "var(--text-secondary)", "fontSize": "0.9rem", "lineHeight": "1.6"}),
                                    html.Hr(style={"borderColor": "var(--card-border)", "margin": "20px 0"}),
                                    html.Div([
                                        html.Strong("Best Performance: ", style={"color": "var(--success)"}),
                                        html.Span(max(metrics_data, key=lambda x: x['dice'])['name'], style={"color": "var(--text-primary)"})
                                    ], style={"marginBottom": "12px"}),
                                    html.Div([
                                        html.Strong("Needs Improvement: ", style={"color": "var(--warning)"}),
                                        html.Span(min(metrics_data, key=lambda x: x['dice'])['name'], style={"color": "var(--text-primary)"})
                                    ])
                                ], style={"padding": "20px"})
                            ], className="graph-container")
                        ], width=12),
                    ])
                ])
                
            except Exception as e:
                print(f"❌ GT processing error: {e}")
                traceback.print_exc()
                gt_tab_content = html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={"fontSize": "2rem", "color": "var(--danger)", "marginBottom": "16px"}),
                    html.H5("Ground Truth Processing Error", style={"color": "var(--text-secondary)", "marginBottom": "8px"}),
                    html.P(f"Error: {str(e)}", style={"color": "var(--text-muted)", "fontSize": "0.9rem"})
                ], style={"textAlign": "center", "padding": "80px 20px"}, className="glass-card")
        
        success_msg = html.Div([
            html.I(className="fas fa-check-circle", style={"marginRight": "8px"}),
            f"Analysis complete! {mask.shape[0]}×{mask.shape[1]}×{mask.shape[2]} volume."
        ], className="status-message status-success")
        
        return (f"{vt:.2f}", f"{vc:.2f}", f"{ve:.2f}", f"{ven:.2f}", fig_ctx, fig_tum, fig_radar, fig_hist, fig_pie, stats_table,
                orig.shape[2] - 1, orig.shape[0] - 1, orig.shape[1] - 1, path, False, {"width": "100%"}, True, success_msg, gt_tab_content)
        
    except Exception as e:
        print(f"❌ DIAGNOSIS ERROR: {e}")
        traceback.print_exc()
        
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
        empty_table = html.Div("No data", style={"color": "var(--text-muted)", "textAlign": "center", "padding": "40px"})
        error_msg = html.Div([html.I(className="fas fa-exclamation-triangle", style={"marginRight": "8px"}), f"Diagnosis failed: {str(e)}"], 
                             className="status-message status-error")
        
        return ("Error", "Error", "Error", "Error", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_table,
                100, 100, 100, None, True, {"width": "0%"}, True, error_msg, empty_gt_tab)


# Global Cache to store loaded brains in RAM
# Format: { 'session_id': { 'img': numpy_array, 'mask': numpy_array } }
DATA_CACHE = {}

def get_slice_fig(path, dim, idx):
    """
    Optimized function to get a slice from RAM if available, 
    otherwise load from disk and cache it.
    """
    if not path: 
        return go.Figure(layout=dict(paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False)))

    try:
        # 1. Use folder name as Session ID
        session_id = os.path.basename(path)

        # 2. Check Cache (RAM) first
        if session_id in DATA_CACHE:
            img = DATA_CACHE[session_id]['img']
            mask = DATA_CACHE[session_id]['mask']
        else:
            # 3. Cache Miss: Load from Disk (Slow, occurs once)
            print(f"⚡ Loading scan into RAM for {session_id}...")
            img_path = os.path.join(path, "img.npy")
            mask_path = os.path.join(path, "mask.npy")
            
            if not os.path.exists(img_path):
                return go.Figure()

            # Load fully into RAM (remove mmap_mode)
            img = np.load(img_path)
            mask = np.load(mask_path)
            
            # Store in Cache
            DATA_CACHE[session_id] = { 'img': img, 'mask': mask }

        # 4. Slice the 3D array based on dimension
        # dim 0 = Sagittal, dim 1 = Coronal, dim 2 = Axial
        if dim == 0: # Sagittal
            img_slice = img[idx, :, :]
            mask_slice = mask[idx, :, :]
        elif dim == 1: # Coronal
            img_slice = img[:, idx, :]
            mask_slice = mask[:, idx, :]
        else: # Axial (Default)
            img_slice = img[:, :, idx]
            mask_slice = mask[:, :, idx]

        # 5. Rotate for correct viewing orientation
        img_slice = np.rot90(img_slice)
        mask_slice = np.rot90(mask_slice)

        # 6. Normalize and Colorize
        max_val = np.max(img_slice)
        img_norm = img_slice / (max_val + 1e-6)
        
        # Create base Grayscale image (R=G=B) + Alpha
        rgba = np.stack([img_norm] * 3 + [np.ones_like(img_norm)], axis=-1) * 255
        
        # Overlay Colors: Red (1), Blue (2), Yellow (3)
        colors = {1: [239, 68, 68], 2: [59, 130, 246], 3: [251, 191, 36]}
        
        for label_id, color in colors.items():
            mask_region = (mask_slice == label_id)
            if np.any(mask_region):
                # Blend: 60% Color + 40% Original Pixel
                for ch in range(3): 
                    rgba[mask_region, ch] = rgba[mask_region, ch] * 0.4 + np.array(color)[ch] * 0.6

        # 7. Create Figure
        fig = go.Figure(go.Image(z=rgba.astype(np.uint8)))
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0), 
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode=False
        )
        return fig

    except Exception as e:
        print(f"❌ SLICE ERROR: {e}")
        return go.Figure(layout=dict(paper_bgcolor="rgba(0,0,0,0)"))

# --- UPDATED CALLBACKS USING THE NEW FUNCTION ---

@app.callback([Output("2d-axial", "figure"), Output("slice-indicator-axial", "children")],
              [Input("slice-slider-axial", "value")], State("results", "data"), prevent_initial_call=True)
def update_axial_slice(idx, path):
    return get_slice_fig(path, 2, idx), f"Slice {idx}"

@app.callback([Output("2d-sagittal", "figure"), Output("slice-indicator-sagittal", "children")],
              [Input("slice-slider-sagittal", "value")], State("results", "data"), prevent_initial_call=True)
def update_sagittal_slice(idx, path):
    return get_slice_fig(path, 0, idx), f"Slice {idx}"

@app.callback([Output("2d-coronal", "figure"), Output("slice-indicator-coronal", "children")],
              [Input("slice-slider-coronal", "value")], State("results", "data"), prevent_initial_call=True)
def update_coronal_slice(idx, path):
    return get_slice_fig(path, 1, idx), f"Slice {idx}"

# DOWNLOAD
@app.callback(Output("dl-nifti", "data"), Input("btn-dl", "n_clicks"), State("results", "data"), prevent_initial_call=True)
def download_mask(n, path):
    if n and path: return dcc.send_file(os.path.join(path, "pred.nii.gz"))
    raise PreventUpdate

# RUN APP
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 NeuroAI Pro Dashboard v4.1 - Optimized with RAM Cache")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Features: Instant Slicing, GT Comparison, Persistent Storage")
    print("=" * 60)
    app.run(debug=True, port=8050, host='0.0.0.0')