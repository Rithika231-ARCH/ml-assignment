"""
Generates a concise, beautifully formatted PDF report for the House Rent Prediction project
using ReportLab.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable

def generate_pdf():
    pdf_filename = "House_Rent_Prediction_Report.pdf"
    
    # 0.5 inch margins = 36 points
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=0,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#4b5563')
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1f2937'),
        alignment=1
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("House Rent Prediction System", title_style))
    story.append(Paragraph("Machine Learning Application & Performance Analysis Report", subtitle_style))
    story.append(Paragraph("<b>Domain:</b> Real Estate Analytics &nbsp;|&nbsp; <b>Model:</b> Gradient Boosting &nbsp;|&nbsp; <b>Dataset:</b> 4,746 Listings (6 Indian Metros)", meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceBefore=2, spaceAfter=8))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This project implements an end-to-end Machine Learning regression pipeline to estimate fair monthly residential rent "
        "across six Indian metropolitan cities (Mumbai, Bangalore, Chennai, Hyderabad, Delhi, and Kolkata). Five regression algorithms "
        "— Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting — were benchmarked. "
        "The <b>Gradient Boosting Regressor</b> achieved the best predictive accuracy with an <b>R² score of 0.8428</b> (5-Fold CV R² = 0.8255) "
        "and a Mean Absolute Percentage Error (MAPE) of <b>29.46%</b>. The <b>Random Forest Regressor</b> achieved the lowest rupee error with a "
        "Mean Absolute Error (MAE) of <b>₹10,609</b>. The trained pipeline was successfully deployed into an interactive Streamlit web application.",
        body_style
    ))
    
    # 2. Problem Statement & Dataset
    story.append(Paragraph("2. Problem Statement & Dataset Overview", h1_style))
    story.append(Paragraph(
        "Residential rental prices exhibit wide variance driven by physical layout, spatial location, and lease conditions. "
        "The objective is to train a supervised machine learning regression model f(x) that maps property attributes to expected monthly rent (INR). "
        "The dataset consists of <b>4,746 validated listings</b> across 12 features with 0 missing values. "
        "Crucial exploratory findings include:",
        body_style
    ))
    story.append(Paragraph("• <b>Right-Skewed Distribution:</b> Median rent is ₹16,000, while the mean is ₹34,993 due to luxury penthouses. Applying log(1 + Rent) normalized the distribution into a near-Gaussian curve.", bullet_style))
    story.append(Paragraph("• <b>Geographic Divergence:</b> Mumbai commands the highest median rent (₹38,000) and mean (₹85,321). Kolkata is the most economical (median ₹9,500), while Bangalore, Chennai, Hyderabad, and Delhi cluster between ₹15,000–₹16,000.", bullet_style))
    story.append(Paragraph("• <b>Size & BHK Impact:</b> Property size (sq. ft.) has a strong non-linear positive correlation with rent.", bullet_style))
    
    # 3. Preprocessing & Feature Engineering
    story.append(Paragraph("3. Preprocessing & Feature Engineering", h1_style))
    story.append(Paragraph(
        "• <b>Floor Decomposition:</b> Parsed composite floor text into continuous variables (<code>Floor_Level</code> and <code>Total_Floors</code>).<br/>"
        "• <b>Domain Ratios:</b> Created <code>Size_per_BHK</code> (room size indicator) and <code>Bathroom_to_BHK_ratio</code>.<br/>"
        "• <b>Encoding & Scaling:</b> Standardized numerical features with <code>StandardScaler</code> and encoded categorical variables (City, Furnishing, Tenant Type, Area Type) via <code>OneHotEncoder(drop='first')</code>.<br/>"
        "• <b>Data Partition:</b> 80% Training (3,796 rows) and 20% Holdout Testing (950 rows) with <code>random_state=42</code>.",
        body_style
    ))
    
    # 4. Model Benchmarking Table
    story.append(Paragraph("4. Machine Learning Model Benchmark", h1_style))
    story.append(Paragraph(
        "All five algorithms were evaluated on the 20% holdout test set and verified using 5-fold cross-validation:",
        body_style
    ))
    
    table_data = [
        [
            Paragraph("<b>Model Name</b>", table_cell_bold),
            Paragraph("<b>MAE (INR) ↓</b>", table_cell_bold),
            Paragraph("<b>RMSE (INR) ↓</b>", table_cell_bold),
            Paragraph("<b>MAPE (%) ↓</b>", table_cell_bold),
            Paragraph("<b>R² (Log) ↑</b>", table_cell_bold),
            Paragraph("<b>5-Fold CV R² ↑</b>", table_cell_bold)
        ],
        [
            Paragraph("Linear Regression (OLS)", table_cell),
            Paragraph("₹ 11,355.71", table_cell),
            Paragraph("₹ 32,935.03", table_cell),
            Paragraph("30.66%", table_cell),
            Paragraph("0.8308", table_cell),
            Paragraph("0.8086 ± 0.009", table_cell)
        ],
        [
            Paragraph("Ridge Regression (L2)", table_cell),
            Paragraph("₹ 11,357.34", table_cell),
            Paragraph("₹ 32,965.38", table_cell),
            Paragraph("30.65%", table_cell),
            Paragraph("0.8308", table_cell),
            Paragraph("0.8086 ± 0.009", table_cell)
        ],
        [
            Paragraph("Decision Tree Regressor", table_cell),
            Paragraph("₹ 12,434.82", table_cell),
            Paragraph("₹ 43,431.38", table_cell),
            Paragraph("33.43%", table_cell),
            Paragraph("0.8028", table_cell),
            Paragraph("0.7801 ± 0.017", table_cell)
        ],
        [
            Paragraph("<b>Random Forest Regressor</b>", table_cell),
            Paragraph("<b>₹ 10,609.37</b>", table_cell_bold),
            Paragraph("₹ 33,949.93", table_cell),
            Paragraph("29.96%", table_cell),
            Paragraph("0.8372", table_cell),
            Paragraph("0.8140 ± 0.008", table_cell)
        ],
        [
            Paragraph("<b>Gradient Boosting (Best)</b>", table_cell),
            Paragraph("₹ 10,795.61", table_cell),
            Paragraph("₹ 36,762.84", table_cell),
            Paragraph("<b>29.46%</b>", table_cell_bold),
            Paragraph("<b>0.8428</b>", table_cell_bold),
            Paragraph("<b>0.8255 ± 0.010</b>", table_cell_bold)
        ],
    ]
    
    t = Table(table_data, colWidths=[130, 75, 75, 65, 70, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#dcfce7')),  # Highlight best model
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    
    # 5. Analysis & Key Insights
    story.append(Paragraph("5. Analysis & Key Drivers", h1_style))
    story.append(Paragraph(
        "• <b>Why Ensembles Won:</b> Gradient Boosting and Random Forest outperformed single trees and linear baselines by capturing complex non-linear interactions between property size, locality, and bathroom count.<br/>"
        "• <b>Target Normalization Effect:</b> Training on raw INR rents yielded an MAE > ₹21,700 due to extreme luxury outliers. Normalizing with log(1 + Rent) cut absolute errors in half to ₹10,600–₹10,795.<br/>"
        "• <b>Top Predictors (Feature Importance):</b> (1) <b>Property Size (sq. ft.)</b> explains >42% of variance; (2) <b>City (Mumbai)</b> adds a substantial spatial premium; (3) <b>Bathroom Count</b> strongly signals luxury tier; (4) <b>Furnishing Status</b> adds an immediate ₹5,000–₹15,000 premium.",
        body_style
    ))
    
    # Add Visual Figures
    if os.path.exists("plots/04_model_comparison.png") and os.path.exists("plots/07_feature_importance.png"):
        story.append(Spacer(1, 4))
        img_table = [
            [
                Image("plots/04_model_comparison.png", width=250, height=95),
                Image("plots/07_feature_importance.png", width=250, height=130)
            ],
            [
                Paragraph("<i>Figure 1: Model Metric Comparison (MAE, RMSE, R²)</i>", table_cell),
                Paragraph("<i>Figure 2: Top Feature Importance Ranking</i>", table_cell)
            ]
        ]
        it = Table(img_table, colWidths=[255, 255])
        it.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(it)
        story.append(Spacer(1, 6))
    
    # 6. Streamlit Web Interface
    story.append(Paragraph("6. Streamlit Web Application Deployment", h1_style))
    story.append(Paragraph(
        "An interactive web application (<code>app.py</code>) was developed using Streamlit to make the model accessible to end-users. "
        "Users can select property specifications (City, BHK, Size, Bathrooms, Floor level, and Furnishing) and receive an instant estimated rent in INR, "
        "a ±15% plausible confidence range, and a dynamic comparison against city median benchmarks. "
        "The application also features dedicated tabs for interactive Exploratory Data Analysis, model diagnostic charts, and project architecture.",
        body_style
    ))
    
    # 7. Conclusion
    story.append(Paragraph("7. Conclusion", h1_style))
    story.append(Paragraph(
        "This project successfully developed and evaluated a reliable machine learning model for residential rental price prediction in urban India. "
        "By combining feature engineering, log-target stabilization, and gradient boosted ensembles, the model achieved an <b>R² of 0.8428</b> and an "
        "<b>MAE of ₹10,795</b>. The system demonstrates practical real-world utility by transforming raw property listings into an automated, transparent valuation tool.",
        body_style
    ))
    
    doc.build(story)
    print(f"PDF successfully generated: {pdf_filename}")

if __name__ == '__main__':
    generate_pdf()
