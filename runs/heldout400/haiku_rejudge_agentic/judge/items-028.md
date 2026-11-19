Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 2 items.

=== ITEM id=67cf5a5a9bc15217 ===
Gold sentences and facts:
Sentence 0: This not only allows the collection of data from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.
  Fact 0.0: It allows the collection of data from complex equipment.
  Fact 0.1: It enables operators to track equipment performance.
  Fact 0.2: It enables operators to predict maintenance.
  Fact 0.3: It enables operators to predict downtimes.
Sentence 1: If a track isn’t used or taken care of for an extended period of time, it will start to wear out and eventually become completely unsuitable for any kind of racing unless it is taken apart and rebuilt.
  Fact 1.0: A track that is not used or taken care of for an extended period of time will start to wear out.
  Fact 1.1: A worn-out track will eventually become completely unsuitable for any kind of racing.
  Fact 1.2: A track becomes unsuitable for racing unless it is taken apart and rebuilt.

Retrieved items:
<item 1>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Considerations for digital twin vendors

3 questions digital twin vendors should ask themselves based on research findings discussed in this article:

1. Sustainability goals: Given the emphasis on sustainability and the role of digital twins in achieving it, is my company’s digital twin solution well-equipped
to support sustainability objectives? Are there new features we need to develop?

2. Interoperability: Is our digital twin solution currently designed to be interoperable with other systems and technologies, especially those from different
vendors?

3. Product development: How aligned is our product development roadmap with the emerging trends highlighted in the report, especially around end-user and
technological advancements?

Considerations for adopters

3 questions adopters should ask themselves based on research findings discussed in this article:

1. Infrastructure readiness: Do we have the necessary infrastructure and resources in place to support the deployment and effective use of digital twins? If
not, what investments are required?

2.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> digital twin market
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, Digital twin market: Analyzing growth and 4 emerging trends, digital twin solution, product development, infrastructure readiness,
sustainability, interoperability
</item 1>

<item 2>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 4: Initiatives promoting interoperability of digital twins across systems from different vendors

Agreement on integration standards among the various digital twin technology providers is crucial for building cross-system digital twins. With standards,
manufacturers can offer services that can be applied to unique situations with other digital twin technology and software.

Recognizing this, countries and industry organizations have taken several steps to address this need:

Plattform Industrie 4.0 and CESMII

Plattform Industrie 4.0 and the Clean Energy Smart Manufacturing Innovation Institute (CESMII) partnered to establish standards for manufacturers in Germany and
the United States to better enable smart, sustainable competition. The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Cell phone shocker as 97% of kids use their device during school hours and beyond, says study -MENTIONS-> software
- Breathe Battery Technologies -MADE-> software
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China

Entities: software, IoT Business News, Plattform Industrie 4.0, CESMII, Industry 4.0, Smart Manufacturing, interoperability of digital twins, integration
standards, digital twins, digital twin technology, Partnership to establish standards for manufacturers, Digital twin market: Analyzing growth and 4 emerging
trends, United States, Germany
</item 2>

<item 3>
Graph query row:
article: Digital twin market: Analyzing growth and 4 emerging trends
publisher: IoT Business News
passages: ['Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nBy the IoT Analytics team.\n\nA new
report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.\n\nFour of these trends are discussed in detail in this
article. These trends are shaping the future of the digital twin market and influencing investment priorities for companies across various industries.\n\nKey
insights:\n\nAccording to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and
2027.\n\n29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital
twins have increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.\n\nThe report notes eight
notable trends helping to advance and promote digital twins, four of which are discussed in this article.', 'Digital twin market: Analyzing growth and 4
emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nDigital twin market snapshot\n\nData from our Digital Twin Market Report 2023–2027 indicates
that 29% of global manufacturing companies have either fully or partially implemented their digital twin strategies, marking a noticeable increase from 20% in
2020. Moreover, the proportion of these companies not contemplating the implementation of digital twins appears to have reduced to 9% in 2023, down from 33.6%
in 2020.\n\nSupporting this trend, an analysis of job postings on SimplyHired suggests a growing demand for digital twin expertise. Postings mentioning “digital
twin” increased by 11% in October 2023 compared to just two years prior, while openings for most other tech topics declined in the same timeframe. Notably,
among the 60 tech-related skill sets we tracked, digital twin-related skills experienced one of the most significant growths.', 'Digital twin market: Analyzing
growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nBased on these observations and additional insights from our report, our current
projection estimates that the global digital twin market could grow at a CAGR of approximately 30% between 2023 and 2027.\n\nAgainst the backdrop of this
growing market, the report notes eight trends that help promote the advancement and adoption of digital twins. Here, we will focus on four:\n\n1. Digital twins
deployed to meet sustainability goals\n\n2. Digital twins employed as virtual sensors in complex conditions\n\n3. Partnerships between cloud hyperscalers and OT
and simulation specialists\n\n4. Initiatives promoting interoperability of digital twins across multifaceted systems\n\nBefore we dive into these trends, and to
provide context, it is helpful to understand how IoT Analytics defines digital twins and the digital twin market.\n\nDefining digital twins and the digital twin
market\n\nIn our March 2023 Decoding Digital Twins article, we define digital twins as a virtual model replicating the behavior of an existing or a potential
real-world asset, system, or multiple systems.', 'Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News,
2023-11-18T16:01:30+00:00)\nThis definition captures a macro concept of digital twins, but our report and previous article present a three-dimensional, cuboid
model to help classify digital twins so apple-to-apple comparisons can be made. Each axis of the cuboid represents one dimension of a digital twin:\n\n1. Life
cycle phase: The X-axis represents the six life cycle phases a digital twin is used for, from design to decommissioning.\n\n2. Hierarchical levels: The Y-axis
represents a digital twin’s five hierarchical levels, from information to multi-system.\n\n3. Use/purpose of implementation: The Z-axis represents the seven
most common uses for digital twins, such as simulation and prediction.\n\nBased on this model, there are 210 potential combinations (5 x 6 x 7 = 210); however,
our research indicates that many digital twin initiatives are tailored to multiple combinations.', 'Digital twin market: Analyzing growth and 4 emerging trends
(Iot Business News, 2023-11-18T16:01:30+00:00)\nIn terms of the digital twin market, our market model only considers software spending for digital twins, which
we break into two scopes:\n\nBroad digital twin market – includes the revenue for all software solutions that provide capabilities that are used for digital
twins, such as solutions that integrate data sources into a digital twin and are capable of simulation, visualization, and predictions\n\n– includes the revenue
for all software solutions that provide capabilities that are used for digital twins, such as solutions that integrate data sources into a digital twin and are
capable of simulation, visualization, and predictions Narrow digital twin market – only considers spending for digital twin-specific software, such as software
intended to model the digital twin data\n\nThough hardware and services related to digital twins can bolster the capabilities of digital twin software, we only
view them as supporting services for the software, and we do not consider them in our digital twin market model. That said, this ability to bolster capabilities
means they can aid digital twin market growth, as shown in some of the trends below.', 'Digital twin market: Analyzing growth and 4 emerging trends (Iot
Business News, 2023-11-18T16:01:30+00:00)\nDigital twin market trends\n\nWe will now look at four trends helping digital twin market growth.\n\nTrend 1: Digital
twins deployed to meet sustainability goals\n\nSustainability was discussed in approximately 21% of recent CEO earnings calls and has remained a consistent
topic throughout 2023. This pairs with our report’s analysis that the pursuit of sustainability goals is a tailwind macro factor for the digital twin
market.\n\nTo achieve their sustainability goals, many companies are exploring digital twins. Due to their ability to simulate real-world conditions and deliver
real-time information, organizations can optimize resource usage, reducing carbon emissions and improving supply and transportation networks.\n\nAccording to
Capgemini Research Institute’s 2022 digital twins report, 57% of organizations agreed that one of the key drivers for their digital twin investments was
improving their sustainability, and 51% agreed that digital twins would help achieve their organization’s environmental sustainability goals.\n\nIn our report,
a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the challenges
inherent in sustainability goals.', 'Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nA digital twin
strategy as part of an overall digitalization plan can be a crucial capability for asset-intensive industries and needs to encompass the entire asset lifecycle,
process, and value chain from design and operations through maintenance and strategic business planning.”\n\nExample: Aden, a Chinese integrated facility
service provider, created a digital twin for one of its commercial centers in Chengdu, China. The digital twin was designed to help facility managers inspect,
maintain, and repair building assets. 3D simulations assist facility managers in visualizing, predicting, and optimizing energy consumption, and the expected
benefits include lower annual energy consumption, water usage, and waste.\n\nTrend 2: Digital twins employed as virtual sensors in complex conditions\n\nVirtual
sensors approximate data that otherwise cannot be obtained via physical sensors, often due to physical sensors being impractical, costly, or hazardous to
employ. Companies are now building digital twins to model hardware and calculate data based on other conditions. This not only allows the collection of data
from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.', 'Digital twin market: Analyzing growth
and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nExample: In large motor applications, such as room-sized motors that pump high volumes of
gas, oil, or other chemicals, operators need to monitor the motors’ temperatures—especially if the motors are high-powered and repeatedly started. If a motor is
too hot during a restart, the components could cause serious damage. Unfortunately, using a direct, physical sensor within a motor is often impractical, and
operators must work from an assumptive time it takes for the motor to cool (with an additional safety buffer time). This means motors can be down longer than
necessary, impacting efficiency and revenue.\n\nTo address this issue, Siemens developed a prototype virtual sensor based on a digital twin. It offers a
simulation of how a physical sensor would operate if it were possible to install it within a motor. Using AR headsets, operators can see a simulation of the
motor and its interior with a demonstrator superimposed over it, and they can see the motor’s temperature.', 'Digital twin market: Analyzing growth and 4
emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nTrend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation
specialists\n\nIn the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure
Digital Twins, that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver
end-to-end digital twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.\n\nHyperscalers
and OT companies\n\nOT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with
hyperscalers, OT companies can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and
computation power.\n\nExample: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.',
'Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nThe companies aimed to accelerate Siemens
Xcelerator adoption and democratize new digital twin solutions using AWS IoT TwinMaker, a service intended to make creating digital twins that incorporate
multiple data sources faster and easier.\n\nHyperscalers and simulation companies\n\nSimulation companies bring specialized expertise in creating accurate,
high-fidelity models of physical entities, be it machinery, buildings, or entire ecosystems. By partnering with a hyperscaler, simulation companies can harness
these cloud services’ data storage and computational power to create more robust, responsive, and accurate data twins that can be scaled and integrated
seamlessly into broader IT ecosystems.\n\nExample: In 2023, Ansys and Microsoft announced a partnership to help customers envision digital twins on a large
scale. In cooperation with Tata Consultancy Services, Ansys and Microsoft integrated Ansys’s Twin Builder’s physics-based simulation capabilities with Azure
Digital Twin and IoT data to allow Twin Builder users to run scenarios and obtain predictions on how their systems will behave.', 'Digital twin market:
Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nTrend 4: Initiatives promoting interoperability of digital twins across
systems from different vendors\n\nAgreement on integration standards among the various digital twin technology providers is crucial for building cross-system
digital twins. With standards, manufacturers can offer services that can be applied to unique situations with other digital twin technology and
software.\n\nRecognizing this, countries and industry organizations have taken several steps to address this need:\n\nPlattform Industrie 4.0 and
CESMII\n\nPlattform Industrie 4.0 and the Clean Energy Smart Manufacturing Innovation Institute (CESMII) partnered to establish standards for manufacturers in
Germany and the United States to better enable smart, sustainable competition. The partnership addressed similar challenges related to Industry 4.0 and Smart
Manufacturing.', 'Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nThe partnership addressed similar
challenges related to Industry 4.0 and Smart Manufacturing.\n\nDTC digital twin interoperability framework\n\nThe Digital Twin Consortium (DTC), a collaborative
partnership of industry, academia, and government experts, released the Digital Twin System Interoperability Framework to “unify a nascent ecosystem of
high-value, multi-vendor services that can seamlessly ‘plug into’ a multi-dimensional, interoperable system of systems.”\n\nDTC and IDTA collaboration\n\nDTC
and the Industrial Digital Twin Association (IDTA) announced a liaison agreement to collaborate on standardization requirements, enabling interoperability
through discussions, aligning work in horizontal domains, and collaborating on open-source projects, contributions, and reference implementations.\n\nDTC and
OPC Foundation collaboration\n\nDTC and the OPC Foundation announced a liaison agreement to “accelerate the development and adoption of digital twin-enabling
technologies,” promoting interoperability standards and processes to advance the use of digital twins in manufacturing across multiple industries.', 'Digital
twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nConsiderations for digital twin vendors\n\n3 questions
digital twin vendors should ask themselves based on research findings discussed in this article:\n\n1. Sustainability goals: Given the emphasis on
sustainability and the role of digital twins in achieving it, is my company’s digital twin solution well-equipped to support sustainability objectives? Are
there new features we need to develop?\n\n2. Interoperability: Is our digital twin solution currently designed to be interoperable with other systems and
technologies, especially those from different vendors?\n\n3. Product development: How aligned is our product development roadmap with the emerging trends
highlighted in the report, especially around end-user and technological advancements?\n\nConsiderations for adopters\n\n3 questions adopters should ask
themselves based on research findings discussed in this article:\n\n1. Infrastructure readiness: Do we have the necessary infrastructure and resources in place
to support the deployment and effective use of digital twins? If not, what investments are required?\n\n2.', 'Digital twin market: Analyzing growth and 4
emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)\nIf not, what investments are required?\n\n2. Data integration: How compatible are our current
data sources and systems with digital twin technologies, especially in terms of sensor data (both physical and virtual/soft)?\n\n3. Interoperability concerns:
As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies and our existing systems?']
connected_entities: ['software', 'Microsoft', 'China', 'digital twin market', 'Digital Twin Market Report 2023–2027', 'SimplyHired', 'global manufacturing
companies', 'digital twin expertise', 'digital twin-related skills', 'digital twins', 'sustainability goals', 'virtual sensors', 'complex conditions', 'cloud
hyperscalers', 'OT', 'simulation specialists', 'multifaceted systems', 'IoT Analytics', 'Decoding Digital Twins', 'three-dimensional, cuboid model', 'life cycle
phase', 'hierarchical levels', 'use/purpose of implementation', 'prediction', 'digital twin initiatives', 'hardware', 'visualization', 'predictions', 'Capgemini
Research Institute', '2022 digital twins report', 'Aden', 'digital twin', 'Chengdu', 'Siemens', 'virtual sensor', 'AR headsets', 'motor', 'physical sensor',
'AWS', 'Microsoft Azure', 'AWS IoT TwinMaker', 'Azure Digital Twins', 'OT companies', 'simulation companies', 'Expansion of the Siemens and AWS partnership',
'digital twin technology', 'industrial data management', 'Siemens Xcelerator', 'Ansys', 'Tata Consultancy Services', 'Twin Builder', 'Azure Digital Twin',
'Ansys and Microsoft partnership', 'Plattform Industrie 4.0', 'CESMII', 'Industry 4.0', 'Smart Manufacturing', 'interoperability of digital twins', 'integration
standards', 'Partnership to establish standards for manufacturers', 'Digital Twin Consortium', 'Digital Twin System Interoperability Framework', 'Industrial
Digital Twin Association', 'OPC Foundation', 'DTC and IDTA liaison agreement', 'DTC and OPC Foundation liaison agreement', 'open-source projects', 'reference
implementations', 'digital twin solution', 'product development', 'infrastructure readiness', 'digital twin technologies', 'data integration', 'sensor data',
'software solutions', 'United States', 'standardization', 'Germany', 'services', 'investments', 'sustainability', 'IoT', 'Simulation', 'manufacturing', 'digital
transformation', 'interoperability']
</item 3>

<item 4>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market trends

We will now look at four trends helping digital twin market growth.

Trend 1: Digital twins deployed to meet sustainability goals

Sustainability was discussed in approximately 21% of recent CEO earnings calls and has remained a consistent topic throughout 2023. This pairs with our report’s
analysis that the pursuit of sustainability goals is a tailwind macro factor for the digital twin market.

To achieve their sustainability goals, many companies are exploring digital twins. Due to their ability to simulate real-world conditions and deliver real-time
information, organizations can optimize resource usage, reducing carbon emissions and improving supply and transportation networks.

According to Capgemini Research Institute’s 2022 digital twins report, 57% of organizations agreed that one of the key drivers for their digital twin
investments was improving their sustainability, and 51% agreed that digital twins would help achieve their organization’s environmental sustainability goals.

In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the
challenges inherent in sustainability goals.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- IoT Analytics -SAID-> digital twins
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, digital twin market, digital twins, Capgemini Research Institute, 2022 digital twins report, Digital twin market: Analyzing growth
and 4 emerging trends, sustainability
</item 4>

<item 5>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> digital twin market
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, Digital twin market: Analyzing growth and 4 emerging trends, digital twin technologies, data integration, sensor data, software
solutions, investments, interoperability
</item 5>

<item 6>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Example: In large motor applications, such as room-sized motors that pump high volumes of gas, oil, or other chemicals, operators need to monitor the motors’
temperatures—especially if the motors are high-powered and repeatedly started. If a motor is too hot during a restart, the components could cause serious
damage. Unfortunately, using a direct, physical sensor within a motor is often impractical, and operators must work from an assumptive time it takes for the
motor to cool (with an additional safety buffer time). This means motors can be down longer than necessary, impacting efficiency and revenue.

To address this issue, Siemens developed a prototype virtual sensor based on a digital twin. It offers a simulation of how a physical sensor would operate if it
were possible to install it within a motor. Using AR headsets, operators can see a simulation of the motor and its interior with a demonstrator superimposed
over it, and they can see the motor’s temperature.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Siemens -PARTICIPATED_IN-> Expansion of the Siemens and AWS partnership
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> digital twin market

Entities: IoT Business News, Siemens, virtual sensor, digital twin, AR headsets, motor, physical sensor, Digital twin market: Analyzing growth and 4 emerging
trends
</item 6>

<item 7>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
By the IoT Analytics team.

A new report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.

Four of these trends are discussed in detail in this article. These trends are shaping the future of the digital twin market and influencing investment
priorities for companies across various industries.

Key insights:

According to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and 2027.

29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital twins have
increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.

The report notes eight notable trends helping to advance and promote digital twins, four of which are discussed in this article.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> SimplyHired
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> global manufacturing companies

Entities: IoT Business News, digital twin market, Digital Twin Market Report 2023–2027, digital twins, IoT Analytics, Digital twin market: Analyzing growth and
4 emerging trends, manufacturing
</item 7>

<item 8>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Based on these observations and additional insights from our report, our current projection estimates that the global digital twin market could grow at a CAGR
of approximately 30% between 2023 and 2027.

Against the backdrop of this growing market, the report notes eight trends that help promote the advancement and adoption of digital twins. Here, we will focus
on four:

1. Digital twins deployed to meet sustainability goals

2. Digital twins employed as virtual sensors in complex conditions

3. Partnerships between cloud hyperscalers and OT and simulation specialists

4. Initiatives promoting interoperability of digital twins across multifaceted systems

Before we dive into these trends, and to provide context, it is helpful to understand how IoT Analytics defines digital twins and the digital twin market.

Defining digital twins and the digital twin market

In our March 2023 Decoding Digital Twins article, we define digital twins as a virtual model replicating the behavior of an existing or a potential real-world
asset, system, or multiple systems.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market -MENTIONS-> sustainability goals
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, digital twin market, digital twins, sustainability goals, virtual sensors, complex conditions, cloud hyperscalers, OT, simulation
specialists, multifaceted systems, IoT Analytics, Decoding Digital Twins, Digital twin market: Analyzing growth and 4 emerging trends, interoperability
</item 8>

<item 9>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation specialists

In the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure Digital Twins,
that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver end-to-end digital
twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.

Hyperscalers and OT companies

OT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with hyperscalers, OT companies
can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and computation power.

Example: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.

Graph facts:
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> AWS
- AWS -MADE-> Machine Learning Toolkit
- Siemens -MADE-> virtual sensor
- A comprehensive list of 2023 tech layoffs -MENTIONS-> AWS
- AWS -PART_OF-> Amazon

Entities: IoT Business News, Siemens, AWS, Microsoft Azure, AWS IoT TwinMaker, Azure Digital Twins, OT companies, simulation companies, Expansion of the Siemens
and AWS partnership, digital twin technology, industrial data management, Digital twin market: Analyzing growth and 4 emerging trends, digital transformation
</item 9>

<item 10>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
A digital twin strategy as part of an overall digitalization plan can be a crucial capability for asset-intensive industries and needs to encompass the entire
asset lifecycle, process, and value chain from design and operations through maintenance and strategic business planning.”

Example: Aden, a Chinese integrated facility service provider, created a digital twin for one of its commercial centers in Chengdu, China. The digital twin was
designed to help facility managers inspect, maintain, and repair building assets. 3D simulations assist facility managers in visualizing, predicting, and
optimizing energy consumption, and the expected benefits include lower annual energy consumption, water usage, and waste.

Trend 2: Digital twins employed as virtual sensors in complex conditions

Virtual sensors approximate data that otherwise cannot be obtained via physical sensors, often due to physical sensors being impractical, costly, or hazardous
to employ. Companies are now building digital twins to model hardware and calculate data based on other conditions. This not only allows the collection of data
from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> China
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> China
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> China
- A comprehensive list of 2023 tech layoffs -MENTIONS-> China
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> China

Entities: China, IoT Business News, virtual sensors, Aden, digital twin, Chengdu, Digital twin market: Analyzing growth and 4 emerging trends
</item 10>
=== END ITEM id=67cf5a5a9bc15217 ===

=== ITEM id=57c489659532b407 ===
Gold sentences and facts:
Sentence 0: Both PMS and mutual funds provide expert guidance in navigating volatile markets and seizing investment opportunities that help create wealth.
  Fact 0.0: PMS provides expert guidance in navigating volatile markets.
  Fact 0.1: PMS provides expert guidance in seizing investment opportunities.
  Fact 0.2: PMS helps create wealth.
  Fact 0.3: Mutual funds provide expert guidance in navigating volatile markets.
  Fact 0.4: Mutual funds provide expert guidance in seizing investment opportunities.
  Fact 0.5: Mutual funds help create wealth.
Sentence 1: Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).
  Fact 1.0: Brown said he was surprised.
  Fact 1.1: Brown was surprised that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results.
  Fact 1.2: Kroll was appointed to consider the reasonableness of Citrin Cooperman's assumptions for HSF's previous set of results.
  Fact 1.3: HSF's previous set of results runs to end of March 2023.

Retrieved items:
<item 1>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Not to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for overseeing the valuation of the
Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”

WHAT’S really going on?

It’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director – is making something
of a ‘hidden’ statement to shareholders with today’s announcement.

It may be a coded attempt by HSF’s board to gnash teeth towards HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the
exclusive power to sweep in and acquire HSF’s portfolio under several different circumstances.

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, valuation, Hipgnosis Songs Fund board, Christopher Mills, Mercuriadis, HSF’s portfolio, music catalogues, call option, Citrin
Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it?, HSM, Music
Business Worldwide
</item 1>

<item 2>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?

Graph facts:
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund

Entities: Rob Naylor, Hipgnosis Songs Fund, Hipgnosis Songs Fund board, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it?, music royalties, Music Business Worldwide, JKBX
</item 2>

<item 3>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 3>

<item 4>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Hipgnosis Songs Fund board, JP Morgan, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it?, Hipgnosis Songs Management, Christopher Brown, Music Business Worldwide
</item 4>

<item 5>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).

Graph facts:
- Taylor Swift is Time’s Person of the Year -MENTIONS-> 1989 (Taylor’s Version)
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> 1989 (Taylor’s Version)
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift

Entities: 1989 (Taylor’s Version), Taylor Swift, Jack Antonoff, Hipgnosis Songs Fund, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin
Cooperman’s latest valuation of the company. What’s it going to do about it?, Jack Antonoff catalog, Lover, Joel Little, Joel Little catalog, Pro-Forma Revenue,
Music Business Worldwide, Merck Mercuriadis
</item 5>

<item 6>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, valuation, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company.
What’s it going to do about it?, Blackstone, Music Business Worldwide, Hipgnosis Song Management, Merck Mercuriadis
</item 6>

<item 7>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, JP Morgan, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of
the company. What’s it going to do about it?, Christopher Brown, Music Business Worldwide
</item 7>

<item 8>
Graph query row:
article: Investment options for high networth investors
mentioned_entities: ['mutual funds', 'risk management', 'India', 'market', 'high-net-worth individuals (HNIs)', 'Alternatives', 'fixed deposits', 'financial and
investment planning', 'Alternative Investment Funds', 'SEBI', 'HNIs', 'public and private markets', 'private equity', 'infrastructure assets', 'venture debt',
'stressed assets', 'start-ups', 'publicly traded equity or debt markets', 'volatility', 'portfolio diversification', 'AIFs', 'high networth investors',
'Start-up funds', 'Angel investors', 'promoters', 'investors', 'Mid-stage private equity (PE) funds', 'mid-stage PE fund', 'late-stage or pre-IPO funds', '2020
stock market rally', 'hedge funds', 'long-only strategies', 'long-short strategies', 'derivative techniques', 'IPOs', 'B2C companies', 'stock exchange',
'annuities', 'funds', 'infrastructure trust', 'capital gains', 'venture debt funds', 'start-up promoters', 'credit', 'special situation funds', 'real estate
yield funds', 'PMS', 'Market-linked Debentures', 'Nifty 50', 'BSE Sensex', '10-year G-Sec', 'equity-linked MLDs', 'principal protection', 'income multipliers',
'MLDs', 'AIF', 'underlying index', 'debenture', 'traditional asset classes', 'risk-return profiles', 'Nuvama Wealth', 'investment options', 'investment
advisors', 'companies', 'stocks', 'banks']
evidence: ['Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)\nIndia’s remarkable economic growth has resulted in a
notable increase in the country’s high-net-worth investors (HNIs). According to recent wealth reports, the population of high-net-worth individuals (HNIs) with
assets worth $1 million or more, which was 7.9 lakh in 2022, will rise to 16.5 lakh, a 107 per cent increase in five years.\n\nThe growing number of HNIs, as
well as their desire to diversify their portfolios and access opportunities outside of traditional avenues such as stocks, mutual funds, and fixed deposits, has
prompted the innovation of existing investment products, leading to the emergence of a new asset class known as ‘Alternatives.’\n\nAlthough the same principles
of financial and investment planning apply to all investors, HNIs have an advantage. They can choose from a broader range of conventional and unconventional
investment opportunities. Investing in unconventional avenues, such as Alternatives, can improve portfolio performance, generate alpha, and allow investors to
grow their wealth faster.\n\nLet us look at a few examples of such investment opportunities:\n\n1.', 'Investment options for high networth investors (Business
Line, 2023-10-28T14:42:28+00:00)\nLet us look at a few examples of such investment opportunities:\n\n1. Alternative Investment Funds (AIFs)\n\nThese funds were
launched in 2012 after the market regulator SEBI issued guidelines for their management. The guidelines enable AIFs to identify and make investment
opportunities available to investors in the public and private markets. AIFs, for example, can invest in listed and unlisted equities via private equity,
infrastructure assets, venture debt, stressed assets, start-ups, and so on.\n\nAs a result, the term “alternatives” came about. The guidelines have divided
these funds into three categories to make it easier for investors to understand the fund’s investment objective. These categories are I, II, and III.\n\nA
noteworthy feature of AIFs is that their returns are generally uncorrelated with publicly traded equity or debt markets, providing a natural hedge against
volatility and genuine portfolio diversification.\n\nBecause of the minimum investment threshold of ₹1 crore, AIFs are only available to HNIs.', 'Investment
options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)\nHowever, not all funds are required to collect ₹1 crore in a single transaction.
They can collect the money in tranches (called drawdowns) over 2-3 years. The drawdown mechanism makes these funds easy on the wallet. AIFs have amassed an AUM
of ₹8.30-lakh crore in the 11 years since their inception, demonstrating the shift of HNI money to this asset.\n\nThe following are some of the investment
opportunities available only to HNIs and presented by AIFs:\n\na) Start-up funds: Angel investors looking to invest in ideas that have the potential to grow
into full-scale businesses can do so through start-up funds, which provide a unique platform for bringing promoters and investors together. These funds handpick
start-ups after extensive due diligence and build a portfolio with the potential to multiply capital several times.\n\nb) Mid-stage private equity (PE) funds:
Investors who are hesitant to take the risk of investing in start-ups can invest in companies in the middle of their evolution cycle.']
</item 8>

<item 9>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
A Final thought…

As mentioned in the opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company
in October, through to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.

Yet, amongst all of the noise, what will likely most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and
BlackRock) is the ongoing underlying commercial performance of the firm’s copyrights.

How are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition strategy holding up against its promise of
growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of the HSF portfolio vs. the multiples that
Merck Mercuriadis paid to acquire it?

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund

Entities: Hipgnosis Songs Fund, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it
going to do about it?, discontinuation vote, Investec Wealth & Investment, Aviva Investors, copyrights, premium-only acquisition strategy, lawsuit, Music
Business Worldwide, BlackRock, Merck Mercuriadis
</item 9>

<item 10>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In response, Hipgnosis Song Management issued its own statement today: “Hipgnosis Song Management has fulfilled its duties to [HSF] with respect to both the
independent valuation and preparation of the interim results in a timely and efficient manner.

“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a constructive manner to support
the interests of the Company and its shareholders.”

The oddest thing about all of this? Citrin Cooperman’s valuation of Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and
overseen… by the HSF board.

See this line, in every annual and interim report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly
Massarsky Consulting, Inc., [is] appointed by the Board to independently value the Company’s Catalogues within the Portfolio.”

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, valuation, Citrin Cooperman Advisors LLC, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest
valuation of the company. What’s it going to do about it?, Music Business Worldwide, Hipgnosis Song Management
</item 10>
=== END ITEM id=57c489659532b407 ===
