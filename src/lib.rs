/*!
# cuda-decision

Decision engine for autonomous agents.

An agent faces hundreds of micro-decisions every second. Which way to go?
What to prioritize? Risk it or play safe? This crate provides the machinery
for consistent, confidence-aware decision making.

- Option generation and ranking
- Multi-criteria utility scoring
- Risk assessment
- Satisficing (good enough > perfect)
- Regret minimization
- Decision trees with confidence propagation
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A decision option
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Option_ {
    pub id: String,
    pub label: String,
    pub criteria: HashMap<String, f64>,  // criterion name → score [0, 1]
    pub confidence: f64,                // how sure we are about these scores
    pub effort: f64,                    // cost to execute
    pub risk: f64,                      // [0, 1] risk level
    pub expected_value: f64,            // calculated
    pub utility: f64,                   // final score
}

/// A criterion for decision making
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Criterion {
    pub name: String,
    pub weight: f64,      // importance [0, 1]
    pub direction: Direction, // higher or lower is better
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction { Higher, Lower }

/// Risk assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub probability: f64,     // [0, 1]
    pub impact: f64,          // [0, 1]
    pub expected_loss: f64,   // probability × impact
    pub category: RiskCategory,
    pub mitigations: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCategory { Low, Medium, High, Critical }

impl RiskAssessment {
    pub fn new(probability: f64, impact: f64) -> Self {
        let expected_loss = probability * impact;
        let category = if expected_loss > 0.7 { RiskCategory::Critical } else if expected_loss > 0.4 { RiskCategory::High } else if expected_loss > 0.15 { RiskCategory::Medium } else { RiskCategory::Low };
        RiskAssessment { probability, impact, expected_loss, category, mitigations: vec![] }
    }
}

/// A decision tree node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionNode {
    pub question: String,
    pub branches: Vec<Branch>,
    pub fallback: Option<String>,  // option id if tree can't decide
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Branch {
    pub condition: String,
    pub outcome: DecisionOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DecisionOutcome {
    Choose(String),   // option id
    Ask(String),      // ask a sub-question → question id
    Defer,            // can't decide now
}

/// Decision history for learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub option_chosen: String,
    pub confidence: f64,
    pub utility: f64,
    pub outcome: f64,  // 0-1 how good the decision was (retrospective)
    pub timestamp: u64,
}

/// The decision engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionEngine {
    pub criteria: Vec<Criterion>,
    pub current_options: Vec<Option_>,
    pub risk_threshold: f64,
    pub satisficing_threshold: f64,
    pub history: Vec<DecisionRecord>,
    pub max_history: usize,
    pub regret_total: f64,
}

impl DecisionEngine {
    pub fn new() -> Self {
        DecisionEngine { criteria: vec![], current_options: vec![], risk_threshold: 0.5, satisficing_threshold: 0.6, history: vec![], max_history: 100, regret_total: 0.0 }
    }

    /// Add a decision criterion
    pub fn add_criterion(&mut self, name: &str, weight: f64, direction: Direction) {
        self.criteria.push(Criterion { name: name.to_string(), weight, direction });
    }

    /// Add an option
    pub fn add_option(&mut self, option: Option_) { self.current_options.push(option); }

    /// Clear options for new decision
    pub fn clear_options(&mut self) { self.current_options.clear(); }

    /// Create a simple option
    pub fn make_option(id: &str, label: &str, scores: HashMap<String, f64>, confidence: f64, effort: f64, risk: f64) -> Option_ {
        Option_ { id: id.to_string(), label: label.to_string(), criteria: scores, confidence, effort, risk, expected_value: 0.0, utility: 0.0 }
    }

    /// Score options using weighted utility
    pub fn evaluate(&mut self) -> Vec<(String, f64)> {
        let total_weight: f64 = self.criteria.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 { return vec![]; }

        for option in &mut self.current_options {
            let mut score = 0.0;
            for criterion in &self.criteria {
                let raw = option.criteria.get(&criterion.name).copied().unwrap_or(0.5);
                let adjusted = match criterion.direction {
                    Direction::Higher => raw,
                    Direction::Lower => 1.0 - raw,
                };
                score += adjusted * (criterion.weight / total_weight);
            }

            // Risk penalty
            if option.risk > self.risk_threshold {
                score *= 1.0 - (option.risk - self.risk_threshold);
            }

            // Effort penalty (slight)
            score *= 1.0 / (1.0 + option.effort * 0.1);

            // Confidence weighting — uncertain options get discounted
            score *= 0.5 + option.confidence * 0.5;

            option.utility = score;
            option.expected_value = score * (1.0 - option.risk * 0.5);
        }

        let mut ranked: Vec<(String, f64)> = self.current_options.iter()
            .map(|o| (o.id.clone(), o.utility))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }

    /// Get best option
    pub fn best(&mut self) -> Option<&Option_> {
        let ranked = self.evaluate();
        ranked.first().and_then(|(id, _)| self.current_options.iter().find(|o| o.id == *id))
    }

    /// Satisficing: return first option above threshold
    pub fn satisfice(&mut self) -> Option<&Option_> {
        self.evaluate();
        self.current_options.iter().find(|o| o.utility >= self.satisficing_threshold)
    }

    /// Assess risk
    pub fn assess_risk(&self, probability: f64, impact: f64) -> RiskAssessment {
        RiskAssessment::new(probability, impact)
    }

    /// Record decision outcome for learning
    pub fn record_outcome(&mut self, option_id: &str, outcome: f64) {
        if let Some(option) = self.current_options.iter().find(|o| o.id == option_id) {
            let regret = (option.utility - outcome).max(0.0);
            self.regret_total += regret;
            self.history.push(DecisionRecord { option_chosen: option_id.to_string(), confidence: option.confidence, utility: option.utility, outcome, timestamp: now() });
            if self.history.len() > self.max_history { self.history.remove(0); }
        }
    }

    /// Cumulative regret
    pub fn average_regret(&self) -> f64 {
        if self.history.is_empty() { return 0.0; }
        self.regret_total / self.history.len() as f64
    }

    /// Did we make a good decision? (retrospective)
    pub fn was_good(&self, option_id: &str, threshold: f64) -> Option<bool> {
        self.history.iter().rev().find(|r| r.option_chosen == option_id).map(|r| r.outcome >= threshold)
    }

    /// Summary
    pub fn summary(&self) -> String {
        format!("DecisionEngine: {} criteria, {} options, {} history, avg_regret={:.3}",
            self.criteria.len(), self.current_options.len(), self.history.len(), self.average_regret())
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_decision() {
        let mut de = DecisionEngine::new();
        de.add_criterion("speed", 0.5, Direction::Higher);
        de.add_criterion("safety", 0.5, Direction::Higher);

        let mut s1 = HashMap::new(); s1.insert("speed".into(), 0.9); s1.insert("safety".into(), 0.3);
        let mut s2 = HashMap::new(); s2.insert("speed".into(), 0.5); s2.insert("safety".into(), 0.8);
        de.add_option(DecisionEngine::make_option("fast", "Fast route", s1, 0.8, 0.2, 0.3));
        de.add_option(DecisionEngine::make_option("safe", "Safe route", s2, 0.9, 0.3, 0.1));

        let ranked = de.evaluate();
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].1 > 0.0);
    }

    #[test]
    fn test_best_option() {
        let mut de = DecisionEngine::new();
        de.add_criterion("quality", 1.0, Direction::Higher);
        let mut s = HashMap::new(); s.insert("quality".into(), 0.9);
        de.add_option(DecisionEngine::make_option("a", "A", s.clone(), 1.0, 0.0, 0.0));
        let mut s2 = HashMap::new(); s2.insert("quality".into(), 0.3);
        de.add_option(DecisionEngine::make_option("b", "B", s2, 1.0, 0.0, 0.0));

        let best = de.best().unwrap();
        assert_eq!(best.id, "a");
    }

    #[test]
    fn test_satisficing() {
        let mut de = DecisionEngine::new();
        de.add_criterion("x", 1.0, Direction::Higher);
        de.satisficing_threshold = 0.5;
        let mut s = HashMap::new(); s.insert("x".into(), 0.6);
        de.add_option(DecisionEngine::make_option("good", "Good enough", s, 1.0, 0.0, 0.0));

        let result = de.satisfice();
        assert!(result.is_some());
    }

    #[test]
    fn test_risk_assessment() {
        let ra = RiskAssessment::new(0.3, 0.3);
        assert_eq!(ra.category, RiskCategory::Medium);
        let ra2 = RiskAssessment::new(0.9, 0.9);
        assert_eq!(ra2.category, RiskCategory::Critical);
    }

    #[test]
    fn test_risk_penalty() {
        let mut de = DecisionEngine::new();
        de.add_criterion("value", 1.0, Direction::Higher);
        de.risk_threshold = 0.3;
        let mut s = HashMap::new(); s.insert("value".into(), 1.0);
        de.add_option(DecisionEngine::make_option("risky", "High value, high risk", s, 1.0, 0.0, 0.9));

        let ranked = de.evaluate();
        // Risk penalty should reduce score significantly
        assert!(ranked[0].1 < 0.6);
    }

    #[test]
    fn test_confidence_discounting() {
        let mut de = DecisionEngine::new();
        de.add_criterion("x", 1.0, Direction::Higher);
        let mut s1 = HashMap::new(); s1.insert("x".into(), 0.8);
        let mut s2 = HashMap::new(); s2.insert("x".into(), 0.8);
        de.add_option(DecisionEngine::make_option("confident", "Sure", s1, 1.0, 0.0, 0.0));
        de.add_option(DecisionEngine::make_option("unsure", "Unsure", s2, 0.2, 0.0, 0.0));

        let ranked = de.evaluate();
        assert_eq!(ranked[0].0, "confident");
    }

    #[test]
    fn test_record_outcome() {
        let mut de = DecisionEngine::new();
        de.add_criterion("x", 1.0, Direction::Higher);
        let mut s = HashMap::new(); s.insert("x".into(), 0.8);
        de.add_option(DecisionEngine::make_option("a", "A", s, 0.8, 0.0, 0.0));
        de.evaluate();
        de.record_outcome("a", 0.9);
        assert_eq!(de.history.len(), 1);
    }

    #[test]
    fn test_regret_tracking() {
        let mut de = DecisionEngine::new();
        de.add_criterion("x", 1.0, Direction::Higher);
        let mut s = HashMap::new(); s.insert("x".into(), 0.9);
        de.add_option(DecisionEngine::make_option("a", "A", s, 0.9, 0.0, 0.0));
        de.evaluate();
        de.record_outcome("a", 0.3); // bad outcome → regret
        assert!(de.average_regret() > 0.0);
    }

    #[test]
    fn test_was_good() {
        let mut de = DecisionEngine::new();
        de.add_criterion("x", 1.0, Direction::Higher);
        let mut s = HashMap::new(); s.insert("x".into(), 0.8);
        de.add_option(DecisionEngine::make_option("a", "A", s, 0.8, 0.0, 0.0));
        de.evaluate();
        de.record_outcome("a", 0.9);
        assert_eq!(de.was_good("a", 0.5), Some(true));
    }

    #[test]
    fn test_direction_lower() {
        let mut de = DecisionEngine::new();
        de.add_criterion("cost", 1.0, Direction::Lower);
        let mut s1 = HashMap::new(); s1.insert("cost".into(), 0.1);
        let mut s2 = HashMap::new(); s2.insert("cost".into(), 0.9);
        de.add_option(DecisionEngine::make_option("cheap", "Cheap", s1, 1.0, 0.0, 0.0));
        de.add_option(DecisionEngine::make_option("expensive", "Expensive", s2, 1.0, 0.0, 0.0));
        let ranked = de.evaluate();
        assert_eq!(ranked[0].0, "cheap");
    }
}
