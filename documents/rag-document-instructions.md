# Instructions for Claude: Creating RAG-Optimized RPG Setting Documents

## Purpose and Context

Claude, when helping to create or update documents for the Starcrash RPG setting, please follow these guidelines to ensure all content is optimized for both human readers and Retrieval Augmented Generation (RAG) systems. These documents will be used in a knowledge base that supports game masters and players.

Your task is to structure content that is semantically rich, consistently formatted, and organized to maximize retrieval accuracy. Always prioritize explicit relationships, controlled vocabulary, and clear hierarchical structure.

## Document Creation Principles

When creating new documents or updating existing ones:

1. **Begin with the standard document structure** (Header, Body, Footer)
2. **Use explicit semantic relationships** rather than implied connections
3. **Apply consistent terminology** from the controlled vocabulary
4. **Structure content hierarchically** from general to specific
5. **Include standardized identifiers** for all named entities
6. **Create concise summaries** at the beginning of each major section
7. **Minimize use of pronouns** in favor of explicit entity references

## Standard Document Structure

### Header Section

Always begin documents with this structure:

```markdown
# [Document Title]

## Document Notes
```yaml
Document Version: 1.0
Version Date: [YYYY-MM-DD]
Collection: [Collection Name]
Tags:
	- [primary_tag]
	- [secondary_tag]
	- [tertiary_tag]
```
```

Guidelines for Header components:
- **Title**: Clear, descriptive, and specific (e.g., "Divine Pantheon: Archosian Order")
- **Version**: Start at 1.0 for new documents
- **Collection**: Group related documents (e.g., "Cosmology", "Geography")
- **Tags**: Include 3-7 specific, relevant keywords for retrieval

### Body Section

Organize body content with:
- A brief (2-3 sentence) summary paragraph at the top
- Hierarchical heading structure (H2, H3, H4)
- YAML blocks for structured data
- Consistent formatting within each section

### Footer Section

End each document with:
```markdown
## Appendices
[Additional supporting information]

## Related Documents
[Cross-references using Document Reference System]
```

## Document Reference Systems

When creating references between documents, implement these three complementary systems:

### 1. Cross-Reference System

Use this format for specific references to content in other documents:

```yaml
Cross References:
    Direct References:
        - "Document Title" (file-name.md): Location in document
        - "Divine Pantheons" (divine-pantheons.md): Section on Archos
        - "Planar Structure" (planes.md): Celestial Axis mechanics
    Key Concepts Referenced:
        - Concept Name (defined in filename.md)
        - Divine Hierarchy (defined in pantheons.md)
        - Planar Boundaries (defined in planes.md)
        - Starcrash Event (defined in timeline.md)
```

### 2. Document Collections

Group documents thematically using this system:

```yaml
Collection Membership:
    Primary Collection: Cosmology
    Related Collections:
        - Divine Entities
        - Planar Mechanics
    Sibling Documents:
        - divine-pantheons.md
        - elemental-planes.md
```

### 3. Document Relationships

Define logical connections between documents:

```yaml
Document Relationships:
    Parent Documents:
        - "Document Name" file-name.md (provides historical context)
    Child Documents:
        - "Archosian Clerics and Worship Practices" archosian-worship.md (expands on specific aspects)
    Lateral Relationships:
        - "The Pantheon of Nef" nef-pantheon.md (connected through twin gods concept)
    External References:
        - D&D Monster Manual (2024 Edition) (pp. 123-124: Related creature statistics)
        - Player's Handbook (2024 Edition) (p. 58: Related class features)
```

## AI Optimization Requirements

### Entity Identification

Apply standardized identifiers for all named entities:

```yaml
Standardized Identifiers:
    - DEI_ARCH_ARCHOS: Primary deity of the Archosian pantheon
    - LOC_MER_TYRCITY: Capital city of the Empire in Meridia 
    - EVT_COS_STARCRASH: The meteoric impact that created magic
```

When mentioning entities in text, include their identifier on first reference:
"Archos [DEI_ARCH_ARCHOS], ruler of the Celestial Axis, established the divine hierarchy..."

### Semantic Relationships

Express relationships explicitly:
- ❌ "The Starcrash and elemental planes." (implied relationship)
- ✅ "The Starcrash [EVT_COS_STARCRASH] caused the creation of six elemental planes."

Use relationship type identifiers:
```yaml
Relationship: 
    Type: causal
    Source: Starcrash Event [EVT_COS_STARCRASH]
    Target: Formation of Elemental Planes [EVT_COS_PLANEFORM]
    Description: "The impact created permanent connections to six elemental demiplanes"
```

### Vector-Friendly Content

For optimal vector embedding:
1. Place distinctive, searchable content early in each section
2. Use consistent terminology for key concepts
3. Limit section length to coherent retrievable chunks (250-500 words)
4. Include a concise summary paragraph at the beginning of each major section

Example:
```markdown
## The Archosian Pantheon [DEI_CAT_ARCHOSIAN]

The Archosian Pantheon consists of thirteen deities organized in a strict hierarchy under Archos [DEI_ARCH_ARCHOS]. This ordered divine structure governs the concepts of law, logic, and structured magic throughout the cosmos.

### Hierarchical Structure
[detailed content here]

### Divine Domains
[detailed content here]
```

## Specific Document Types

When creating specific document types, include these standard sections:

### Location Documents
```yaml
Required Sections:
    - Geographic Summary (2-3 sentences)
    - Geographic Details (coordinates, terrain, climate)
    - Cultural Information (inhabitants, customs, language)
    - Political Structure (government, relations)
    - Notable Features (landmarks, resources)
    - Historical Significance (key events)
    - Game Mechanics (relevant rules, hooks)
```

### Character/Deity Documents
```yaml
Required Sections:
    - Entity Summary (2-3 sentences)
    - Core Identity (nature, purpose, alignment)
    - Appearance (physical description)
    - Lore (history, myths)
    - Relationships (connections to other entities)
    - Worshippers/Followers (if applicable)
    - Doctrine (beliefs, tenets)
    - Game Mechanics (domains, powers, stats)
```

### Historical Documents
```yaml
Required Sections:
    - Event Summary (2-3 sentences)
    - Timeline (chronological sequence)
    - Key Figures (participants)
    - Causes and Effects (explicit relationships)
    - Cultural Impact (lasting influence)
    - Related Events (with relationship types)
    - Game Implications (how it affects play)
```

## Controlled Vocabulary

Maintain consistency by using these preferred terms:

```yaml
Controlled Vocabulary:
    Entity Types:
        - deity | god | divine being (use "deity" consistently)
        - plane | realm | dimension (use "plane" consistently)
        - spell | magical effect | arcane working (use "spell" consistently)
    Relationships:
        - causes | results in | leads to (use "causes" consistently)
        - contains | houses | includes (use "contains" consistently)
        - rules | governs | controls (use "rules" consistently)
    Actions:
        - created | formed | made (use "created" consistently)
        - destroyed | demolished | ruined (use "destroyed" consistently)
        - transformed | changed | altered (use "transformed" consistently)
    States:
        - active | functioning | operational (use "active" consistently)
        - dormant | inactive | quiescent (use "dormant" consistently)
        - destroyed | ruined | obliterated (use "destroyed" consistently)
```

## Example Document Structure

Here's how a complete document should be structured:

```markdown
# The Divine Pantheon of Archos

## Document Notes
```yaml
Document Version: 1.0
Version Date: 2025-03-02
Collection: Cosmology
Tags:
	- deities
	- archosian
	- order
	- divine hierarchy
```

The Archosian Pantheon represents order, law and structure in the cosmos. Formed after the Birth of Twin Gods, this pantheon established the fundamental principles of cosmic order and created the first hierarchical divine organization.

## Pantheon Overview [DEI_CAT_ARCHOSIAN]

The Archosian Pantheon consists of thirteen deities organized in a strict hierarchy under Archos [DEI_ARCH_ARCHOS]. This ordered divine structure governs the concepts of law, logic, and structured magic throughout the cosmos.

### Hierarchical Structure
```yaml
Divine Hierarchy:
    Supreme Deity: Archos [DEI_ARCH_ARCHOS]
    Greater Deities:
        - Mabus [DEI_ARCH_MABUS]: Logic and Reason
        - Kraxus [DEI_ARCH_KRAXUS]: Justice and Order
    Lesser Deities:
        - [Additional deities listed]
```

### Historical Development
```yaml
Key Events:
    Formation:
        Date: Approximately 90,000 YA
        Cause: Created by Archos [DEI_ARCH_ARCHOS] 
        Process: Collection of primordial remains infused with divine essence
    Major Transformations:
        - Divine War [EVT_DIV_WAR]: Led to restructuring after betrayal of Kraxus
        - Divine Withdrawal [EVT_DIV_WITHDRAW]: Changed relationship with material plane
```

## Appendices

### Divine Symbols
[Symbol descriptions]

### Sacred Texts
[Text descriptions]

## Related Documents
```yaml
Cross References:
    Direct References:
        - "Timeline" (timeline.md): Birth of Twin Gods section
        - "Planes" (planes.md): Celestial Axis section
    Key Concepts Referenced:
        - Divine Hierarchy (defined here)
        - Twin Gods (defined in timeline.md)

Collection Membership:
    Primary Collection: Cosmology
    Related Collections:
        - Divine Entities
        - Planar Mechanics

Document Relationships:
    Parent Documents:
        - "The Cosmic History of the Elder Plane" cosmic-history.md (provides historical context)
    Child Documents:
        - "Archosian Clerics and Worship Practices" archosian-worship.md (expands on specific aspects)
    Lateral Relationships:
        - "The Pantheon of Nef" nef-pantheon.md (connected through twin gods concept)
    External References:
        - Player's Handbook (2024 Edition) (p. 58: Law domain mechanics)
```
```

## Quality Assurance Requirements

Before finalizing any document, ensure it meets these criteria:

```yaml
Structural Requirements:
    - All YAML sections properly formatted with consistent indentation
    - No duplicate keys or inconsistent spacing
    - All required sections included for document type

Content Requirements:
    - Entity identifiers included for all named entities
    - Relationships explicitly stated rather than implied
    - Controlled vocabulary terms used consistently
    - Cross-references accurate and specific

RAG Optimization Requirements:
    - Each document section viable as independent retrieval chunk
    - Key concepts defined directly in document
    - Section summaries included for major sections
    - Distinctive content placed early in each section
```
