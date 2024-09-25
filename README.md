# College Wealth Wizard

The College Wealth Wizard is a Multi-Agent LLM System. The Wizard is intended for use by current and prospective college students, college graduates considering a career change, as well as college advisors providing advice and counseling to their students. The objective of the Wizard is to give users a transparent idea of how much they will pay for college and how much they can expect to make in their current or prospective career. The Wizard presents information in new ways, such as major-university combinations with net positive and net negative investments, as well as college rankings for the highest paying major-career combinations.  

![CWW_Diagram.png](data%2FCWW_Diagram.png)

## Agents
### Agent I: RAG with CIP-SOC Code Database
**Source 1:** Bureau of Labor Statistics  
**Source 2:** National Center for Education Statistics  

### Agent II: Urban Institute Education Data Portal API
The Urban Institute Education Data Portal API provides information on student aid applicant characteristics, campus-based awards, title IV grants, post college earnings by institution, and federal loan repayment and default rates and averages.

### Agent III: College Scorecard API
The College Scorecard API provides information on field of study by institution, typical monthly loan payments, post-graduation employment rates, time to degree, and Pell grants. 

### Agent IV: Internet Search API
The Tavily Internet Search API is a search engine optimized for LLMs and RAG that can be used to retrieve resources from the internet. 

## Audience
- Prospective college students
- College graduates considering a career change
- College advisors 