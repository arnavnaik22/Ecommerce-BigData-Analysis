#!/bin/bash

echo "================================================================================"
echo "    E-COMMERCE BIG DATA ANALYSIS - EXECUTION STARTING"
echo "================================================================================"
echo ""

# Run the Spark Job and save output to log
spark-submit final_ultra.py 2>&1 | tee complete_output.log

echo ""
sleep 2
clear
echo ""

echo "================================================================================"
echo "              E-COMMERCE BIG DATA ANALYSIS - EXECUTIVE SUMMARY                  "
echo "================================================================================"
echo ""

echo "--------------------------------------------------------------------------------"
echo "  EXECUTION METRICS"
echo "--------------------------------------------------------------------------------"
grep "Total time:" complete_output.log | sed 's/^/  /'
grep "Records:" complete_output.log | sed 's/^/  /'
grep "Models trained:" complete_output.log | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  CONVERSION FUNNEL ANALYSIS"
echo "--------------------------------------------------------------------------------"
grep "Views:" complete_output.log | sed 's/^/  /'
grep "Add to Cart:" complete_output.log | sed 's/^/  /'
grep "Purchases:" complete_output.log | sed 's/^/  /'
grep "Conversion Rate:" complete_output.log | sed 's/^/  /'
grep "Cart Abandonment:" complete_output.log | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  TOP PERFORMING BRAND"
echo "--------------------------------------------------------------------------------"
grep "Top Brand:" complete_output.log | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  CUSTOMER SEGMENTATION (RFM ANALYSIS)"
echo "--------------------------------------------------------------------------------"
grep -A 5 "Customer Segments:" complete_output.log | tail -5 | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  RDD OPERATIONS SUMMARY"
echo "--------------------------------------------------------------------------------"
grep "Purchase events:" complete_output.log | sed 's/^/  /'
grep "Unique products:" complete_output.log | sed 's/^/  /'
grep "Total events:" complete_output.log | sed 's/^/  /'
grep "Event breakdown:" complete_output.log | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  MACHINE LEARNING MODEL PERFORMANCE"
echo "--------------------------------------------------------------------------------"
grep "AUC-ROC:" complete_output.log | sed 's/^/  /'
grep "Accuracy:" complete_output.log | sed 's/^/  /'
grep "Precision:" complete_output.log | sed 's/^/  /'
grep "Recall:" complete_output.log | sed 's/^/  /'
grep "F1-Score:" complete_output.log | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  FEATURE IMPORTANCE RANKING (TOP 5)"
echo "--------------------------------------------------------------------------------"
grep -A 5 "Feature Importance:" complete_output.log | tail -5 | sed 's/^/  /'
echo ""

echo "--------------------------------------------------------------------------------"
echo "  OUTPUT FILES GENERATED"
echo "--------------------------------------------------------------------------------"
ls -lh ecommerce_analysis_dashboard.png complete_output.log 2>/dev/null | awk '{printf "  %-45s %10s\n", $9, $5}'
echo ""
