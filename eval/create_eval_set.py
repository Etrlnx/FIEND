#!/usr/bin/env python
"""Create evaluation dataset for financial RAG benchmarking."""

import json
from pathlib import Path


def load_existing_eval_set(path: str = "eval/eval_set.json"):
    with open(path) as f:
        return json.load(f)


def create_expanded_eval_set():
    """Create expanded evaluation set with 87+ questions across categories."""

    with open("eval/eval_set.json") as f:
        existing = json.load(f)

    additional = [
        # Factual
        {
            "id": "eval_46",
            "category": "factual",
            "ticker": "MSFT",
            "form": "10-Q",
            "question": "What are Microsoft's primary cloud service offerings as described in its 10-Q?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Azure", "cloud services", "server products", "enterprise services"],
            "expected_answer": "Microsoft's cloud services include Azure, server products, and enterprise services.",
        },
        {
            "id": "eval_47",
            "category": "factual",
            "ticker": "GOOGL",
            "form": "10-Q",
            "question": "What are Google's main advertising products?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Google Search", "YouTube ads", "Google Network", "Google Ad Manager"],
            "expected_answer": "Google's advertising products include Google Search, YouTube ads, Google Network, and Google Ad Manager.",
        },
        {
            "id": "eval_48",
            "category": "factual",
            "ticker": "AMZN",
            "form": "10-Q",
            "question": "What are Amazon's principal business activities?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["e-commerce", "AWS", "cloud computing", "digital streaming", "artificial intelligence"],
            "expected_answer": "Amazon's principal business activities include e-commerce, AWS cloud computing, digital streaming, and AI services.",
        },
        {
            "id": "eval_49",
            "category": "factual",
            "ticker": "META",
            "form": "10-Q",
            "question": "What are Meta's primary sources of revenue?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["advertising", "Family of Apps", "Reality Labs"],
            "expected_answer": "Meta's primary revenue sources are advertising through Family of Apps and Reality Labs.",
        },
        {
            "id": "eval_50",
            "category": "factual",
            "ticker": "NVDA",
            "form": "10-Q",
            "question": "What are NVIDIA's reportable segments?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Compute & Networking", "Graphics"],
            "expected_answer": "NVIDIA's reportable segments are Compute & Networking and Graphics.",
        },
        {
            "id": "eval_51",
            "category": "factual",
            "ticker": "TSLA",
            "form": "10-K",
            "question": "What is Tesla's mission statement according to its 10-K?",
            "target_sections": ["Item 1"],
            "ground_truth_keywords": ["accelerate", "world's transition", "sustainable energy"],
            "expected_answer": "Tesla's mission is to accelerate the world's transition to sustainable energy.",
        },
        {
            "id": "eval_52",
            "category": "factual",
            "ticker": "JPM",
            "form": "10-K",
            "question": "What are JPMorgan Chase's core values?",
            "target_sections": ["Item 1"],
            "ground_truth_keywords": ["client service", "integrity", "excellence", "diversity"],
            "expected_answer": "JPMorgan Chase's core values include client service, integrity, excellence, and diversity.",
        },
        {
            "id": "eval_53",
            "category": "factual",
            "ticker": "GS",
            "form": "10-K",
            "question": "What is Goldman Sachs' business model?",
            "target_sections": ["Item 1"],
            "ground_truth_keywords": ["investment banking", "securities", "investment management", "consumer banking"],
            "expected_answer": "Goldman Sachs operates in investment banking, securities, investment management, and consumer banking.",
        },
        {
            "id": "eval_54",
            "category": "factual",
            "ticker": "BAC",
            "form": "10-K",
            "question": "What are Bank of America's responsible growth principles?",
            "target_sections": ["Item 1"],
            "ground_truth_keywords": ["responsible growth", "customer focus", "risk management", "technology"],
            "expected_answer": "Bank of America's responsible growth principles focus on customer focus, risk management, and technology innovation.",
        },
        # More numerical
        {
            "id": "eval_55",
            "category": "numerical",
            "ticker": "MSFT",
            "form": "10-K",
            "question": "What was Microsoft's total revenue for fiscal year 2026?",
            "target_sections": ["Item 7", "Item 8"],
            "ground_truth_keywords": ["revenue", "2026", "2025"],
            "expected_answer": "Microsoft's total revenue for fiscal year 2026.",
        },
        {
            "id": "eval_56",
            "category": "numerical",
            "ticker": "GOOGL",
            "form": "10-K",
            "question": "What was Alphabet's operating income for 2025?",
            "target_sections": ["Item 7", "Item 8"],
            "ground_truth_keywords": ["operating income", "2025", "2024"],
            "expected_answer": "Alphabet's operating income for 2025.",
        },
        {
            "id": "eval_57",
            "category": "numerical",
            "ticker": "AMZN",
            "form": "10-K",
            "question": "What was Amazon's net income for 2025?",
            "target_sections": ["Item 7", "Item 8"],
            "ground_truth_keywords": ["net income", "2025", "2024"],
            "expected_answer": "Amazon's net income for 2025.",
        },
        {
            "id": "eval_57",
            "category": "numerical",
            "ticker": "META",
            "form": "10-K",
            "question": "What was Meta's earnings per share for 2025?",
            "target_sections": ["Item 7", "Item 8"],
            "ground_truth_keywords": ["earnings per share", "EPS", "2025", "2024"],
            "expected_answer": "Meta's earnings per share for 2025.",
        },
        {
            "id": "eval_58",
            "category": "numerical",
            "ticker": "NVDA",
            "form": "10-Q",
            "question": "What was NVIDIA's gross margin for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["gross margin", "margin", "percentage"],
            "expected_answer": "NVIDIA's gross margin for the most recent quarter.",
        },
        # More temporal
        {
            "id": "eval_59",
            "category": "temporal",
            "ticker": "AAPL",
            "form": "10-K",
            "question": "How did Apple's Services revenue change from fiscal 2024 to 2025?",
            "target_sections": ["Item 7", "Item 8"],
            "ground_truth_keywords": ["Services", "revenue", "2025", "2024", "growth"],
            "expected_answer": "Apple's Services revenue change from fiscal 2024 to 2025.",
        },
        {
            "id": "eval_60",
            "category": "temporal",
            "ticker": "MSFT",
            "form": "10-K",
            "question": "How did Microsoft's Azure revenue growth rate change year-over-year?",
            "target_sections": ["Item 7", "MD&A"],
            "ground_truth_keywords": ["Azure", "revenue growth", "year-over-year", "percentage"],
            "expected_answer": "Microsoft's Azure revenue growth rate change year-over-year.",
        },
        {
            "id": "eval_60",
            "category": "temporal",
            "ticker": "GOOGL",
            "form": "10-K",
            "question": "How did Google Cloud revenue change from 2024 to 2025?",
            "target_sections": ["Item 7", "MD&A"],
            "ground_truth_keywords": ["Google Cloud", "revenue", "2025", "2024", "growth"],
            "expected_answer": "Google Cloud revenue change from 2024 to 2025.",
        },
        # More cross-document
        {
            "id": "eval_61",
            "category": "cross-document",
            "ticker": "AUTO",
            "form": "10-K",
            "question": "How do Tesla, Ford, and GM describe electric vehicle transition risks?",
            "target_sections": ["Item 1A", "Risk Factors"],
            "ground_truth_keywords": ["electric vehicle", "transition", "risk", "competition", "regulation"],
            "expected_answer": "All three automakers cite EV transition risks including competition, regulation, and technology shifts.",
        },
        {
            "id": "eval_61",
            "category": "cross-document",
            "ticker": "ENERGY",
            "form": "10-K",
            "question": "How do Exxon, Chevron, and ConocoPhillips describe climate transition risks?",
            "target_sections": ["Item 1A", "Risk Factors"],
            "ground_truth_keywords": ["climate", "transition", "carbon", "regulation", "energy transition"],
            "expected_answer": "All three energy companies cite climate transition risks including regulation and energy transition.",
        },
        {
            "id": "eval_62",
            "category": "cross-document",
            "ticker": "RETAIL",
            "form": "10-K",
            "question": "How do Walmart, Target, and Costco describe supply chain risks?",
            "target_sections": ["Item 1A", "Risk Factors"],
            "ground_truth_keywords": ["supply chain", "disruption", "logistics", "inventory", "tariff"],
            "expected_answer": "All three retailers cite supply chain disruption risks including logistics and inventory management.",
        },
        # More unanswerable
        {
            "id": "eval_63",
            "category": "unanswerable",
            "ticker": "MSFT",
            "form": "10-K",
            "question": "What will be Microsoft's stock price on December 31, 2030?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question. SEC filings do not provide speculative stock price forecasts.",
        },
        {
            "id": "eval_62",
            "category": "unanswerable",
            "ticker": "GOOGL",
            "form": "10-K",
            "question": "What is the exact number of Google Search queries per day in 2027?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        {
            "id": "eval_62",
            "category": "unanswerable",
            "ticker": "XYZ",
            "form": "10-K",
            "question": "What was AcmeCorp's revenue in 2025?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question. No filings exist for AcmeCorp.",
        },
        {
            "id": "eval_63",
            "category": "unanswerable",
            "ticker": "AAPL",
            "form": "10-K",
            "question": "What is the chemical composition of Apple's M4 chip?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        {
            "id": "eval_63",
            "category": "unanswerable",
            "ticker": "NVDA",
            "form": "10-Q",
            "question": "What is NVIDIA's internal roadmap for GPU architecture through 2030?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        # Table-dependent
        {
            "id": "eval_64",
            "category": "numerical",
            "ticker": "MSFT",
            "form": "10-Q",
            "question": "What was Microsoft's Intelligent Cloud revenue for the quarter ended March 31, 2026?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Intelligent Cloud", "revenue", "revenue"],
            "expected_answer": "Microsoft's Intelligent Cloud revenue for the quarter.",
            "requires_table": True
        },
        {
            "id": "eval_63",
            "category": "numerical",
            "ticker": "GOOGL",
            "form": "10-Q",
            "question": "What was Google Cloud's operating income for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Google Cloud", "operating income", "operating"],
            "expected_answer": "Google Cloud's operating income for the most recent quarter.",
            "requires_table": True
        },
        {
            "id": "eval_62",
            "category": "numerical",
            "ticker": "AMZN",
            "form": "10-Q",
            "question": "What was AWS revenue for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["AWS", "revenue", "revenue"],
            "expected_answer": "AWS revenue for the most recent quarter.",
            "requires_table": True
        },
        {
            "id": "eval_62",
            "category": "numerical",
            "ticker": "META",
            "form": "10-Q",
            "question": "What was Family of Apps revenue for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Family of Apps", "revenue", "revenue"],
            "expected_answer": "Family of Apps revenue for the most recent quarter.",
            "requires_table": True
        },
        {
            "id": "eval_63",
            "category": "numerical",
            "ticker": "NVDA",
            "form": "10-Q",
            "question": "What was Data Center revenue for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["Data Center", "revenue", "revenue"],
            "expected_answer": "Data Center revenue for the most recent quarter.",
            "requires_table": True
        },
        # More table-dependent
        {
            "id": "eval_64",
            "category": "numerical",
            "ticker": "TSLA",
            "form": "10-Q",
            "question": "What was Tesla's automotive gross margin for the most recent quarter?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["automotive", "gross margin", "margin"],
            "expected_answer": "Tesla's automotive gross margin for the most recent quarter.",
            "requires_table": True
        },
        {
            "id": "eval_65",
            "category": "numerical",
            "ticker": "JPM",
            "form": "10-Q",
            "question": "What was JPMorgan's CET1 capital ratio?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["CET1", "capital ratio", "ratio"],
            "expected_answer": "JPMorgan's CET1 capital ratio.",
            "requires_table": True
        },
        {
            "id": "eval_66",
            "category": "numerical",
            "ticker": "BAC",
            "form": "10-Q",
            "question": "What was Bank of America's net interest margin?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["net interest margin", "margin", "percentage"],
            "expected_answer": "Bank of America's net interest margin.",
            "requires_table": True
        },
        {
            "id": "eval_66",
            "category": "numerical",
            "ticker": "WMT",
            "form": "10-Q",
            "question": "What was Walmart's e-commerce sales growth rate?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["e-commerce", "growth", "percentage"],
            "expected_answer": "Walmart's e-commerce sales growth rate.",
            "requires_table": True
        },
        {
            "id": "eval_67",
            "category": "numerical",
            "ticker": "COST",
            "form": "10-Q",
            "question": "What was Costco's comparable sales growth?",
            "target_sections": ["Item 1", "Item 2"],
            "ground_truth_keywords": ["comparable sales", "growth", "percentage"],
            "expected_answer": "Costco's comparable sales growth.",
            "requires_table": True
        },
        # More unanswerable
        {
            "id": "eval_67",
            "category": "unanswerable",
            "ticker": "TSLA",
            "form": "10-K",
            "question": "What will be Tesla's Model Y production volume in 2028?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question. SEC filings do not provide speculative multi-year forward production forecasts.",
        },
        {
            "id": "eval_68",
            "category": "unanswerable",
            "ticker": "JPM",
            "form": "10-K",
            "question": "What will be the Federal Reserve interest rate in 2027?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question. SEC filings do not provide speculative economic forecasts.",
        },
        {
            "id": "eval_68",
            "category": "unanswerable",
            "ticker": "GOOGL",
            "form": "10-K",
            "question": "What is the secret algorithm for Google Search ranking?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        {
            "id": "eval_69",
            "category": "unanswerable",
            "ticker": "AMZN",
            "form": "10-K",
            "question": "What is Jeff Bezos's personal net worth in 2026?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        {
            "id": "eval_69",
            "category": "unanswerable",
            "ticker": "META",
            "form": "10-K",
            "question": "What is Mark Zuckerberg's private email address?",
            "target_sections": [],
            "ground_truth_keywords": [],
            "expected_answer": "Insufficient evidence to answer this question.",
        },
        # Ambiguous/contradictory
        {
            "id": "eval_70",
            "category": "ambiguous_contradictory",
            "ticker": "MSFT",
            "form": "10-K",
            "question": "Did Microsoft's gaming revenue grow or decline compared to Sony's PlayStation revenue?",
            "target_sections": ["Item 1A", "Item 7"],
            "ground_truth_keywords": ["gaming", "revenue", "Xbox", "PlayStation"],
            "expected_answer": "Insufficient evidence to answer this question. The evidence contains Microsoft's gaming revenue but not a direct comparison with Sony's PlayStation revenue.",
        },
        {
            "id": "eval_71",
            "category": "ambiguous_contradictory",
            "ticker": "AUTO",
            "form": "10-K",
            "question": "Which automaker has better EV margins: Tesla, Ford, or GM?",
            "target_sections": ["Item 1A", "Item 7"],
            "ground_truth_keywords": ["EV", "margin", "margin"],
            "expected_answer": "Insufficient evidence to answer this question. The evidence contains each company's EV revenue but not a direct margin comparison.",
        },
    ]

    return additional


def main():
    with open("eval/eval_set.json") as f:
        existing = json.load(f)

    additional = create_additional_questions()

    all_questions = existing + additional

    output_path = Path("eval/eval_set.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_questions, indent=2), encoding="utf-8")

    print(f"Expanded eval set: {len(all_questions)} questions")
    cats = {}
    for q in all_questions:
        cats[q['category']] = cats.get(q['category'], 0) + 1
    print('Categories:', cats)
    print(f"Table-dependent: {sum(1 for q in all_questions if q.get('requires_table'))}")


if __name__ == "__main__":
    main()