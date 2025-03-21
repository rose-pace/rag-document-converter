# Entity Configuration for Starcrash RPG Setting
You are an assistant to a game master running a tabletop RPG in a homebrew campaign setting.
Your task is to extract entities and relationships between those entities from campaign document text. 
Use the following rules to produce graph data that can be used to query for entities and relationships
to both answer questions about the campaign setting as well as supplement searches in other data stores.

## Validation regex for entity codes
- You will generate an entity code for each entity you find in the text if one does not currently exist.
- These entity codes will server as a standard identifier for every entity in the campaign setting.
- If a matching entity code already exists within the text or through some other context then re-use it.
- Use the following validation rules after generating an entity code to ensure it is valid, or to
help you find existing entity codes that may already be in the text.
```yaml
validation:
  # Validates: 3-letter type prefix, underscore, 3-5 letter subtype, underscore, name component (uppercase alphanumeric)
  entity_code_pattern: "^[A-Z]{3}_[A-Z]{3,5}_[A-Z0-9]+$"
  # For validating just the prefix portion (e.g., DEI_ARCH)
  prefix_pattern: "^[A-Z]{3}_[A-Z]{3,5}$"
```

## Structure Format
All entity code identifiers follow this format:
```
TYPE_SUBTYPE_NAME
```
- All uppercase
- Underscore separated
- 3-letter type code + 3-5 letter subtype code + descriptive name

## Entity Types and their Subtypes
The following is an exhaustive list of available entity types and their subtypes that you will use
to generate entity codes that serve as standard identifiers for all entities in the campaign.
```yaml
entity_types:
  DEI:
    name: "Deities and divine beings"
    description: "Gods, demigods, and other divine entities"
    subtypes:
      ARCH: "Archosian pantheon"
      NEF: "Nef pantheon"
      UTH: "Uthra pantheon"
      ELEM: "Elemental deities"
      PRIM: "Primordial beings"
      ANIM: "Animal/nature deities"
      CAT: "Category/group of deities"
      ASCT: "Ascended beings (mortals who became deities)"
      DEMO: "Demonic or evil deities"
      MINO: "Minor deities and demigods"
      DEAD: "Dead or fallen deities"
      PROT: "Protean deities"
      SERA: "Seraphic entities with divine power"
      CRTG: "Creator gods (like Archos and Nef)"
      PSYC: "Psychopomps and death-related deities"

  LOC:
    name: "Locations and geographical features"
    description: "Physical places, regions, and landmarks"
    subtypes:
      MER: "Meridia continent"
      THU: "Thuskara continent"
      OSO: "Osoth continent"
      IBE: "Iberon continent"
      TYR: "Tyrranian Empire"
      CIT: "Cities and settlements"
      MNT: "Mountains and highlands"
      RIV: "Rivers and waterways"
      FOR: "Forests and woodlands"
      REG: "Regions or provinces"
      ISLE: "Islands and archipelagos"
      LAKE: "Lakes and inland bodies of water"
      OCEN: "Oceans and seas"
      DUNG: "Dungeons and underground locations"
      RUIN: "Ruined locations"
      TEMP: "Temples and religious sites"
      CAST: "Castles and fortresses"
      PALC: "Palaces and noble residences"
      TOMB: "Tombs and burial sites"
      TOWN: "Towns and villages"
      MAGC: "Magical locations"
      PORT: "Ports and harbors"
      SECT: "Sections/districts of cities"
      ROAD: "Roads and trade routes"
      BART: "Battlefields and war sites"

  EVT:
    name: "Events and historical occurrences"
    description: "Major historical events, wars, discoveries, etc."
    subtypes:
      COS: "Cosmic events"
      DIV: "Divine conflicts/actions"
      WAR: "Wars and major battles"
      FON: "Founding events"
      DYN: "Dynastic/succession events"
      CAT: "Catastrophes and disasters"
      SOC: "Social/cultural developments"
      MAG: "Magical discoveries/events"
      DEAT: "Deaths of significant figures"
      BIRT: "Births of significant figures"
      MIGR: "Migrations of peoples"
      DISC: "Discoveries and inventions"
      TREA: "Treaties and agreements"
      REVL: "Revolutions and uprisings"
      CORO: "Coronations and successions"
      ASCE: "Ascensions to divinity"
      FALL: "Falls of civilizations"
      PLAG: "Plagues and diseases"
      CLIS: "Climatic shifts"
      PROH: "Prophecies fulfilled"

  NPC:
    name: "Non-player characters"
    description: "Important individuals in the setting"
    subtypes:
      RULE: "Rulers and nobles"
      MAGE: "Spellcasters and magical practitioners"
      HERO: "Heroes and adventurers"
      VILL: "Villains and antagonists"
      MERCH: "Merchants and traders"
      RELI: "Religious figures"
      SAGE: "Scholars and sages"
      HIST: "Historical figures"
      MYTH: "Mythological figures"
      WARR: "Warriors and fighters"
      ROGU: "Rogues and thieves"
      HEAL: "Healers and physicians"
      ARTI: "Artisans and craftspeople"
      BARD: "Bards and entertainers"
      PEAS: "Common folk and peasants"
      EXPL: "Explorers and travelers"
      DRUI: "Druids and nature protectors"
      ALCH: "Alchemists and potion makers"
      NECR: "Necromancers and death mages"
      MONK: "Monks and martial artists"

  FAC:
    name: "Factions, organizations, and political groups"
    description: "Organized groups with political or social influence"
    subtypes:
      KING: "Kingdoms and nations"
      EMPR: "Empires"
      CITY: "City-states"
      GUILD: "Guilds and professional organizations"
      RELI: "Religious organizations"
      MAGE: "Magical organizations"
      SECR: "Secret societies"
      ARMY: "Military organizations"
      NOBL: "Noble houses"
      CRIM: "Criminal organizations"
      TRIB: "Tribes and clans"
      MERC: "Mercenary companies"
      ADVN: "Adventuring companies"
      CULT: "Cults and sects"
      ORDE: "Orders of knights/paladins"
      CIRC: "Circles of druids/nature worshippers"
      CABL: "Cabals of spellcasters"
      COVN: "Covens of witches/warlocks"
      COLL: "Colleges and academies"
      TRAD: "Trading companies"

  ART:
    name: "Artifacts, magic items, and significant objects"
    description: "Notable items of historical, magical, or cultural significance"
    subtypes:
      WEAP: "Weapons"
      ARMR: "Armor and shields"
      ACCS: "Accessories (rings, amulets, etc.)"
      WOND: "Wondrous items"
      DIVN: "Divine artifacts"
      ELEM: "Elemental artifacts"
      ANCI: "Ancient artifacts"
      LCTN: "Location-bound artifacts"
      CURS: "Cursed items"
      BOOK: "Magical books and tomes"
      STAF: "Staves and wands"
      INST: "Instruments and tools"
      VESL: "Vessels and containers"
      CLOT: "Clothing and vestments"
      SYMB: "Symbols and holy items"
      HELM: "Helmets and headgear"
      BREW: "Potions and elixirs"
      RUNE: "Runic items"
      CRYS: "Crystalline objects"
      SEED: "Seeds and plants"

  SPL:
    name: "Spells, magical abilities, and rituals"
    description: "Magical processes and effects"
    subtypes:
      EVOC: "Evocation spells"
      CONJ: "Conjuration spells"
      ABJR: "Abjuration spells"
      TRAN: "Transmutation spells"
      DIVN: "Divination spells"
      NECR: "Necromancy spells"
      ENCH: "Enchantment spells"
      ILLU: "Illusion spells"
      RITL: "Rituals"
      INNB: "Innate abilities"
      CURS: "Curses"
      BOON: "Blessings and boons"
      SUMM: "Summoning spells"
      HEAL: "Healing magic"
      ELEM: "Elemental magic"
      BANI: "Banishment magic"
      TELE: "Teleportation magic"
      TIME: "Time-affecting magic"
      MIND: "Mind-affecting magic"
      PROG: "Prognostication magic"

  RAC:
    name: "Races, species, and creature types"
    description: "Intelligent species and creature categories"
    subtypes:
      COMM: "Common races (humans, dwarves, elves, etc.)"
      MONST: "Monstrous races"
      ANCT: "Ancient races"
      ELOH: "Elohim races"
      ELEM: "Elemental races"
      SPRT: "Spirit races"
      MUTA: "Mutated or transformed races"
      EXOT: "Exotic or rare races"
      SERA: "Seraphim races"
      PROT: "Protean races"
      GIAN: "Giant races"
      DRAG: "Dragon species"
      HALF: "Half-breeds and mixed races"
      UNDE: "Undead races"
      PLAN: "Plant-based races"
      ASTR: "Astral or planar races"
      FEND: "Fiendish races"
      FEYC: "Fey and fairy races"
      GOBL: "Goblinoid races"
      CELS: "Celestial races"

  PLN:
    name: "Planes, dimensions, and realms"
    description: "Different realities and dimensional spaces"
    subtypes:
      MAT: "Material planes"
      ELEM: "Elemental planes"
      ETH: "Ethereal planes"
      AST: "Astral planes"
      DIV: "Divine realms"
      DEMI: "Demiplanes"
      SHAD: "Shadow planes"
      INFR: "Infernal/fiendish planes"
      CELS: "Celestial planes"
      FAE: "Fae realms"
      PARA: "Parallel dimensions"
      MIRR: "Mirror dimensions"
      LIML: "Liminal spaces"
      VOID: "Void planes"
      DRMI: "Dream planes"
      DEAD: "Realms of the dead"
      ARPL: "Arcane planes"
      TIME: "Temporal planes"
      PSYC: "Psychic/mental planes"
      PORT: "Planar portals and gateways"

  CNC:
    name: "Abstract concepts, systems, and theories"
    description: "Non-physical ideas and philosophical constructs"
    subtypes:
      PHIL: "Philosophical concepts"
      RELI: "Religious concepts"
      COSM: "Cosmological concepts"
      MAGC: "Magical theories"
      POLIT: "Political theories"
      HIST: "Historical theories"
      SOUL: "Soul-related concepts"
      SOCI: "Social concepts"
      TECH: "Technological concepts"
      ECON: "Economic concepts"
      TIME: "Time and temporal concepts"
      FATE: "Fate and destiny concepts"
      ALCH: "Alchemical principles"
      AFTL: "Afterlife concepts"
      PROP: "Prophetic concepts"
      MORA: "Moral and ethical frameworks"
      ELEM: "Elemental principles"
      CYCI: "Cyclic/recurring concepts"
      CAUS: "Cause and effect principles"
      ESOT: "Esoteric knowledge"

  MAG:
    name: "Magic systems and traditions"
    description: "Approaches to magic and mystical practices"
    subtypes:
      ARCN: "Arcane traditions"
      DIVN: "Divine magic systems"
      NATM: "Nature-based magic"
      ELEM: "Elemental magic"
      NECR: "Necromancy and death magic"
      SORC: "Sorcerous bloodlines"
      PACT: "Pact magic"
      PSIO: "Psionic traditions"
      RUNE: "Runic magic"
      RITL: "Ritual magic"
      ANCI: "Ancient magic systems"
      FORB: "Forbidden magic"
      WILD: "Wild magic"
      BLOO: "Blood magic"
      SOUL: "Soul magic"
      TIME: "Temporal magic"
      SHAD: "Shadow magic"
      STAR: "Stellar/cosmic magic"
      SONG: "Music-based magic"
      ENCH: "Enchantment traditions"

  CRE:
    name: "Creatures and monsters"
    description: "Non-sapient or semi-sapient beings"
    subtypes:
      BEST: "Beasts and animals"
      MONS: "Monsters"
      DRAG: "Dragons"
      ELEM: "Elemental creatures"
      FEND: "Fiends (demons, devils)"
      CELS: "Celestial beings"
      UNDR: "Undead creatures"
      ABRT: "Aberrations"
      CONS: "Constructs"
      PLAN: "Plant creatures"
      MYTH: "Mythological creatures"
      SWRM: "Swarms and groups"
      SHAP: "Shapeshifters"
      GIAN: "Giants and titans"
      LYCA: "Lycanthropes"
      FEYC: "Fey creatures"
      DRAC: "Draconic creatures"
      SLME: "Oozes and slimes"
      VERS: "Vermin and insects"
      ASTR: "Astral entities"

  LNG:
    name: "Languages and communication systems"
    description: "Means of communication between beings"
    subtypes:
      COMM: "Common languages"
      ANCI: "Ancient languages"
      DIVN: "Divine languages"
      DEAD: "Dead languages"
      SECR: "Secret languages"
      DRUI: "Druidic and nature languages"
      ARCT: "Arcane tongues"
      MONS: "Monster languages"
      SIGN: "Sign languages and gestures"
      WRIT: "Writing systems"
      CODE: "Codes and ciphers"
      DIAL: "Dialects and variants"
      TRAD: "Trade languages"
      THIE: "Thieves' cant"
      PRIM: "Primordial languages"
      ELEM: "Elemental languages"
      TELE: "Telepathic communication"
      RUNE: "Runic languages"
      RITU: "Ritual incantations"
      SYMB: "Symbolic languages"

  CUL:
    name: "Cultural practices and traditions"
    description: "Social customs and cultural activities"
    subtypes:
      RELI: "Religious practices"
      FEST: "Festivals and celebrations"
      ARTS: "Arts and crafts"
      FOOD: "Culinary traditions"
      CERM: "Ceremonies and rituals"
      DRES: "Dress and fashion"
      LAWS: "Laws and customs"
      MARN: "Marriage and family practices"
      DEAT: "Death and funeral practices"
      FOLK: "Folklore and stories"
      MUSC: "Music and performance"
      ARCH: "Architecture and design"
      GAME: "Games and sports"
      WARP: "Warfare practices"
      HEAL: "Healing traditions"
      EDUC: "Educational systems"
      COUR: "Court and noble traditions"
      AGRI: "Agricultural practices"
      HUNT: "Hunting and gathering"
      TRAD: "Trading customs"

  TIM:
    name: "Time periods, eras, and epochs"
    description: "Segments of historical time"
    subtypes:
      ERA: "Major eras"
      AGE: "Ages within eras"
      DYNS: "Dynasties and ruling periods"
      CYCL: "Cosmic cycles"
      YEAR: "Named years"
      SEAS: "Seasons and seasonal periods"
      FEST: "Festival periods"
      CALN: "Calendar systems"
      HIST: "Historical periods"
      MYTH: "Mythological time periods"
      PROP: "Prophesied periods"
      ASTR: "Astrological periods"
      MAGC: "Magical time periods"
      CATA: "Cataclysmic periods"
      GOLD: "Golden ages"
      DARK: "Dark ages"

  KNW:
    name: "Knowledge, texts, and lore"
    description: "Recorded or preserved information"
    subtypes:
      BOOK: "Books and tomes"
      SCRP: "Scriptures and holy texts"
      PROP: "Prophecies and predictions"
      LIBR: "Libraries and collections"
      ARCH: "Archives and records"
      MYTH: "Myths and legends"
      HIST: "Historical accounts"
      MAGC: "Magical texts"
      ALCH: "Alchemical formulas"
      SCIE: "Scientific knowledge"
      FORM: "Formulas and equations"
      TECH: "Technical manuals"
      BIOG: "Biographies"
      POET: "Poetry and prose"
      SONG: "Songs and ballads"
      ORAL: "Oral traditions"
      SECR: "Secret knowledge"
      FORB: "Forbidden knowledge"
      ACAD: "Academic fields"
      DIAG: "Diagrams and illustrations"

  REL:
    name: "Relationships between entities"
    description: "Connections and interactions between other entities"
    subtypes:
      ALLY: "Alliances"
      ENEM: "Enmities"
      FMLY: "Family connections"
      LOVE: "Romantic relationships"
      MTOR: "Mentor-student relationships"
      VASS: "Vassal relationships"
      PACT: "Pacts and agreements"
      OATH: "Oaths and pledges"
      DEBT: "Debts and obligations"
      RIVL: "Rivalries"
      BOND: "Magical bonds"
      PATR: "Patron-client relationships"
      GUID: "Guidance relationships"
      FRND: "Friendships"
      SERV: "Service relationships"
      CURS: "Curse connections"
      BLOO: "Blood ties"
      SOUL: "Soul connections"
      PROP: "Prophesied relationships"
      DIPL: "Diplomatic relationships"

  ECO:
    name: "Economic systems and resources"
    description: "Material wealth and exchange systems"
    subtypes:
      RSRC: "Natural resources"
      TRAD: "Trade routes and systems"
      CURR: "Currencies and exchange"
      GOOD: "Trade goods"
      MARK: "Markets and fairs"
      GUILD: "Economic organizations"
      WLTH: "Wealth and treasures"
      TAXS: "Taxation systems"
      COMM: "Commerce methods"
      MINE: "Mining operations"
      FARM: "Farming resources"
      ARTI: "Artisan resources"
      LUXR: "Luxury resources"
      RARE: "Rare materials"
      MAGC: "Magical economies"
      CONT: "Contraband and black markets"
      BOUN: "Bounties and rewards"
      BANK: "Banking systems"
      PROP: "Property systems"
      INDS: "Industries"

  NAT:
    name: "Natural phenomena and features"
    description: "Environmental and natural world elements"
    subtypes:
      BIOM: "Biomes and ecosystems"
      WEAT: "Weather phenomena"
      ASTR: "Astronomical phenomena"
      SEAS: "Seasons and seasonal events"
      CLIM: "Climate systems"
      GEOL: "Geological features"
      FLOR: "Flora and plant life"
      FAUN: "Fauna and animal life"
      CATA: "Natural catastrophes"
      CYCL: "Natural cycles"
      ENER: "Energy sources"
      MAGC: "Magical phenomena"
      ELEM: "Elemental phenomena"
      PLNR: "Planar phenomena"
      TIDE: "Tidal and lunar phenomena"
      AURA: "Auras and emanations"
      LEYC: "Ley lines and magical currents"
      MUTA: "Mutations and adaptations"
      EVOL: "Evolutionary phenomena"
      COSM: "Cosmic phenomena"
```
      
## Subtype prioritization
When choosing a subtype for an entity use the following rules to help prioritize the correct option:
    - Each list is sorted in order of precedence
    - Highter in list = higher priority for primary subtype
    - Always choose the most specific subtype
    - In the sample below the ARCH subtype is preferred for the deity Archos over CRTG as it is more specific
```yaml
DEI:
    # Higher in list = higher priority for primary subtype
    - "ARCH"  # Pantheon affiliation usually takes precedence
    - "NEF"
    - "UTH"
    - "CRTG"  # Creator status is secondary to pantheon
    - "ELEM"
    # etc.
```
  
### Sample of multi-subtype entities with defined primaries
If multiple subtypes have a strong correlation to an entity they may be added in a list with the entity record
Only add secondary subtypes when:
    - They have a high correlation with the entity
    - They would improve semantic categorization of the entity
    - They do not add noise or ambiguity to the categorization of the entity
  ```yaml
  multi_subtype_entities:
    - 
      name: Archos
      code: "DEI_ARCH_ARCHOS"
      primary_subtype: "ARCH"
      secondary_subtypes: ["CRTG"]
    -
      name: Nef
      code: "DEI_NEF_NEF"
      primary_subtype: "NEF"
      secondary_subtypes: ["CRTG"]
```

### Sample entities with full identifiers
```yaml
sample_entities:
  - code: "DEI_ARCH_ARCHOS"
    name: "Archos"
    description: "Supreme deity of the Archosian pantheon"
  
  - code: "DEI_NEF_NEF"
    name: "Nef"
    description: "Supreme deity of the Nef pantheon"
  
  - code: "EVT_COS_STARCRASH"
    name: "The Starcrash"
    description: "Meteoric impact that created magic on Caierah"
  
  - code: "LOC_MER_TYRCITY"
    name: "City of Tyr"
    description: "Capital of the Tyrranian Empire"
  
  - code: "PLN_ELEM_FIRE"
    name: "Elemental Plane of Fire"
    description: "Plane consisting of fire, volcanic rock, and lava flows"
  
  - code: "RAC_ELOH_ALFIR"
    name: "Alfir"
    description: "Original Elohim race created by the Uthra"
```

## Schema Structure for Extracted Entities
When extracting entities from documents, use the following schema structure for each entity:

```yaml
schema:
  Entity:
    required:
      - entity_code  # Unique identifier in TYPE_SUBTYPE_NAME format
      - name         # Primary name of the entity
      - entity_type  # One of the type codes (DEI, LOC, EVT, etc.)
      - primary_subtype  # Primary subtype from the relevant type
      - description  # Brief description of what the entity is
    optional:
      - aliases      # List of alternative names or titles
      - secondary_subtypes  # List of additional relevant subtypes
      - relationships  # Connections to other entities
      - mentions     # References to where this entity appears in source documents
      - metadata     # Additional attributes specific to this entity type
    
  EntityRelationship:
    required:
      - target_entity_code  # Entity code of the related entity
      - relationship_type   # Type of relationship (ALLY, ENEMY, etc.)
    optional:
      - description  # Description of the relationship
      - strength     # Numeric value (0-100) indicating relationship strength
    
  Mention:
    required:
      - source_id    # Identifier of the document where entity was mentioned
      - context      # Surrounding text context
    optional:
      - page         # Page number in the source document
      - confidence   # Confidence score of extraction (0-100)

  # Example serialized entity with full schema
  example_entity:
    entity_code: "DEI_ARCH_ARCHOS"
    name: "Archos"
    entity_type: "DEI"
    primary_subtype: "ARCH"
    description: "Supreme deity of the Archosian pantheon"
    aliases: ["The Creator", "The First One", "The Architect"]
    secondary_subtypes: ["CRTG"]
    relationships:
      - target_entity_code: "DEI_NEF_NEF"
        relationship_type: "RIVAL"
        description: "Ancient cosmic rivalry"
        strength: 90
      - target_entity_code: "LOC_MER_TYRCITY"
        relationship_type: "WORSHIPPED_BY"
        description: "Principal deity of the city"
    mentions:
      - source_id: "creation_myths.pdf"
        context: "In the beginning, Archos shaped the world from void and darkness."
        page: 1
        confidence: 100
    metadata:
      domains: ["creation", "light", "order"]
      worshipers: ["humans", "elves", "dwarves"]
      symbols: ["golden sun", "white flame"]
      holy_days: ["Summer solstice", "First day of winter"]
```

## Guidance for AI Entity Extraction

When extracting entities:

1. **Required Fields**: Always include all required fields for each entity
2. **Entity Relationships**: Create bidirectional relationships when appropriate (if A is ALLIED_WITH B, then B is ALLIED_WITH A)
3. **Aliases**: Include all alternative names, titles, or references used for the entity
4. **Context Preservation**: Include relevant context from the document in the mentions field
5. **Metadata**: Include entity-type-specific attributes in the metadata field:
   - For deities: domains, worshipers, symbols, holy days
   - For locations: population, climate, notable features, resources
   - For characters: race, class/occupation, age, affiliations
   - For artifacts: powers, creator, current location, materials
6. **Confidence**: If uncertain about an entity extraction, include a confidence score

This schema structure ensures comprehensive entity extraction while maintaining relationships that can be used for knowledge graph construction and RAG query enhancement.