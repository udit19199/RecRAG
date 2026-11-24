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

=== ITEM id=eb0513802241f702 ===
Gold sentences and facts:
Sentence 0: “The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the environment,” Goodyear SVP of global operations and CTO Chris Helsel says.
  Fact 0.0: 6PPD-quinone was identified in late 2020
  Fact 0.1: 6PPD-quinone requires more research to understand what impact, if any, it may have on aquatic life or the environment
  Fact 0.2: Chris Helsel is Goodyear's SVP of global operations and CTO
  Fact 0.3: Chris Helsel says the 6PPD-quinone statement
Sentence 1: In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the challenges inherent in sustainability goals.
  Fact 1.0: The report contains a quote from a former VP at an industrial automation vendor
  Fact 1.1: The former VP at an industrial automation vendor said digital solutions provide visibility, analysis, and insight
  Fact 1.2: The digital solutions address the challenges inherent in sustainability goals

Retrieved items:
<item 1>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Scholars have argued that putting too much emphasis on the platforms can be too simplistic and reeks of technological determinism; they have also pointed out
that cultures like the United States’ and the legacy media have a long history with post-truth. That said, there are certainly dynamics going on that we did not
anticipate, and we don’t seem quite sure what to do with them, even with multiple areas of scholarship in communication, disinformation, and social media and
democracy working on these inquiries for years.

What seems clear is that the Internet is not the connectedness we imagined. The ecology and spirituality of the 1960s, which shaped and structured much of what
we see as energy democracy and the good future today, told us we were all connected. Globally networked — it sounds familiar, like a fevered dream from the
1980s or 1990s, a dream that in turn had its roots in the 1960s and before.
</item 1>

<item 2>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Authoritarians know this, and this is why they have troll farms that can manipulate the range of solutions and the sentiments about them.

The Internet as we experience it represents a central obstacle to climate action, through several mechanisms. Promotion of false information about climate
change is only one of them. There’s general political polarization, which inhibits the coalitions we need to build to realize clean energy, as well as creates
paralyzing infighting within the climate movement about strategies, which the platforms benefit from. There’s networked opposition to the infrastructure we need
for the energy transition. There’s the constant distraction from the climate crisis, in the form of the churning scandals of the day, in an attention economy
where all topics compete for mental energy. And there’s the drain of time and attention spent on these platforms rather than in real-world actions.

Any of these areas are worth spending time on, but this essay focuses on how the contemporary media ecology interferes with climate strategy and infrastructure
in particular.
</item 2>

<item 3>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
In terms of the digital twin market, our market model only considers software spending for digital twins, which we break into two scopes:

Broad digital twin market – includes the revenue for all software solutions that provide capabilities that are used for digital twins, such as solutions that
integrate data sources into a digital twin and are capable of simulation, visualization, and predictions

– includes the revenue for all software solutions that provide capabilities that are used for digital twins, such as solutions that integrate data sources into
a digital twin and are capable of simulation, visualization, and predictions Narrow digital twin market – only considers spending for digital twin-specific
software, such as software intended to model the digital twin data

Though hardware and services related to digital twins can bolster the capabilities of digital twin software, we only view them as supporting services for the
software, and we do not consider them in our digital twin market model. That said, this ability to bolster capabilities means they can aid digital twin market
growth, as shown in some of the trends below.
</item 3>

<item 4>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This will include optimizing the production process of hydrogels as well as sourcing affordable raw materials and minimizing waste generated during the
manufacturing of these materials.

“Ensuring the safety and reliability of hydrogel-based batteries is also necessary,” Fan concluded. “Rigorous testing and evaluation of the potential
environmental impact of hydrogel electrolytes, from raw material extraction to disposal, are vital to meeting sustainability goals and gaining public trust.”

Reference: Peihua Yang, Hong Jin Fan, et al., Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Advanced Materials,
(2023). DOI: 10.1002/adma.202306531

Feature image credit: danilo.alvesd on Unsplash
</item 4>

<item 5>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Product design is in a moment of profound change and redefinition as technologies like artificial intelligence (AI) and spatial computing dramatically affect
computing experiences. AI, especially, may have only minor impacts on interface design but will significantly impact the holistic product or ecosystem
experience. Spatial computing, on the other hand, will change human-computer interactions and unmoor our understanding of what a computer is.

In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for
technologists and designers to be in the process together.

For successful products and businesses, innovation is perpetual. There is a never-ending search to find the next new thing that enhances the user experience,
extends product reach, expands revenue, or all three simultaneously. Product design makes innovation less daunting and increases chances for success because it
is a multidisciplinary process with structures and frameworks to catalyze innovation. Technologists have a role in the process that expands beyond simply
validating the technology or concept, which is their customary responsibility.
</item 5>

<item 6>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“For the next five years, if you want to sell tires in California, you need to go through the process of either eliminating 6PPD, or at least properly
researching the alternatives,” Molden says.

“We are collaborating with researchers and other scientists to better understand this transformation product, fill knowledge gaps, and determine next steps,”
says Helsel.

Other initiatives are coming to limit overall tire emissions, including the upcoming Euro 7 regulations, which put caps on tire wear particles as well as other
emissions.

That seems like progress, but Emissions Analytics’ Molden isn’t so sure: “The reason I’m worried about that is I’m sure you can reduce that wear mass, but if
you do that by making the compound more toxic, you could actually make the environmental impact worse.”

“You could actually make the environmental impact worse.”

That could be akin to increasing a car’s fuel efficiency by switching to leaded fuel.

Molden believes that what’s needed is better understanding.
</item 6>

<item 7>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
In a new paper published in the journal Advanced Materials, Nanyang Technological University (NTU), School of Physical and Mathematical Sciences professor
Hongjin Fan and his co-authors examine different electrolytes for aqueous zinc ion batteries that take the form of hydrogels — three-dimensional network
structures that are capable of storing large amounts of water — to address some of these challenges.

“The core concept of our research is that we design the hydrogel electrolytes by introducing functional groups [specific groupings of atoms within molecules
that have their own characteristic properties] that carry either positive or negative charges,” Fan said. “Our idea is to use a hetero-hydrogel to mitigate
challenges on both the anode and the cathode sides of the zinc iodine batteries.”

A hydrogel boost

Fan explained that the purpose of this hydrogel modification is to ensure that when zinc accumulates on the anode side, it does so evenly.
</item 7>

<item 8>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
For most technical leads and software developers, this is a mode inversion from our traditional tactical and direct interaction with technology. The context
switch from daily build-and-operate is challenging but is paramount to developing successful and innovative products. We are uniquely positioned to generate
strategic insights translated from dense technical minutiae that drive innovative business cases and product experiences.

A technology innovation must solve a business problem, such as improving operational efficiency, growing existing revenue streams, or generating new ones. The
problem space may be customer-facing (e.g., how can we deliver a new feature?) or internal-facing (e.g., how can we make a process more efficient?). The problem
is the paramount concern. The specific technologies or innovations used to solve the problem are often less critical. We cannot lose perspective of the business
needs; otherwise, the activities become too academic or a paid hobby.

An ordinary domestic analogy is hanging a picture. The hole size, bracket, or tools used to hang the picture are immaterial as long as the picture is on the
wall and straight.
</item 8>

<item 9>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
If giants like Ford are feeling the frustration, imagine the mood at a small green tech startup.

So what’s the play? How can the government support those striving against an increasingly insecure (and insincere) backdrop?

The response of any tech company impacted by last week’s news surely has to be this.

First, we need a consistent macroeconomic policy. This has a major impact on startups raising and deploying capital into climate R&D. The global economic
contraction has made it difficult for these businesses to raise any money. Overall approaches to the economy have trickle-down effects on how we, as businesses,
make money and run our operations. Right down to salaries, which alone are hard to keep up with inflation-based pay rises.

The second is a consistent tax policy. One of the most critical things for startups is research and development (R&D) tax credits.
</item 9>

<item 10>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
New research aims to improve the stability and safety of alternatives to rechargeable lithium-ion batteries using aqueous zinc and hydrogels.

The demand for rechargeable batteries as a result of the growth of both the personal electronics and electric vehicle markets has grown steadily over the last
decade. Lithium-ion batteries are one of the technologies that have emerged and matured due to this demand.

In 2020, Statista estimated that lithium-ion batteries accounted for 185 gigawatts (GWh) of energy consumed, and this energy demand is projected to reach 2,035
GWh by 2030. Most of this growth is accounted for by the increased desire for electric vehicles as people turn away from fossil fuels.

While it is undeniable that lithium-ion batteries are more environmentally friendly than fossil fuels and while they are also relatively inexpensive, these
batteries aren’t perfect.
</item 10>

<item 11>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
The thing we’re jacked into and collectively creating seems more like a global endocrine system than anything we might have visualized in the years while
“cyber” was a prefix. This may seem a banal observation, given that Marshall McLuhan was talking about the global nervous system more than fifty years ago. We
had enthusiasm about cybernetics and global connectivity over the decades and, more recently, a revitalization of theory about networks and kinship and rhizomes
and all the rest. (The irony is that with fifty years of talk on “systems thinking,” we still have responses to things like COVID-19 or climate that are almost
antithetical to considering interconnected systems — dominated by one set of expertise and failing to incorporate the social sciences and humanities). So —
globally connected, yet divided into silos, camps, echo-chambers, and so on. Social media platforms are acting as agents, structuring our interactions and our
spaces for dialogue and solution-building.
</item 11>

<item 12>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The government calls for embedding resilience, sustainability, and nature-based solutions in infrastructure planning and investment to enhance climate
resilience. Likewise, investments in renewable energy are expected not only to enhance energy security but also reduce greenhouse gas (GHG) emissions.

From an economic vantage point, GHG emissions are economic “bads” and are a cause of inefficiency as they get over-produced. Their effects, however, go beyond
national borders, and while the Philippines contributes minimally to global GHG emissions, the World Risk Index 2022 report identifies the Philippines as the
most disaster-prone country in the world.

The government envisions that by 2028: (1) Climate and disaster risk resilience of communities and institutions will increase (2) Ecosystem resilience will be
enhanced, and (3) A low-carbon economy transition is enabled.

To ensure that these are realized, the government is set to safeguard cross-sectoral convergence and implement a comprehensive risk management approach to
address the adverse consequences posed by climate change.
</item 12>

<item 13>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
It relies on modern composites instead of air pressure to provide the right mix of stiffness and flex with significantly more durability. Michelin says it could
reduce tire replacements by 20 percent. Variations of this theme have been in development for decades, but models are in testing now for consumer and industrial
use.

Goodyear gave us a more radical vision of a more distant future tire with 2020’s ReCharge concept, which proposed that a biodegradable tread could be recharged
simply by inserting refillable capsules with seasonal compounds. As a concept, it’s pretty far out there. Goodyear’s Helsel confesses as much, calling them
“purely conceptual designs” that “might never be produced as such.” However, he says that the non-pneumatic nature of those concepts, and the advanced sensors
embedded within, are indicative of ongoing research into future tires.

Another concept, though, is less about increasing the sustainability of tires and more about increasing the viability of EVs.
</item 13>

<item 14>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?
</item 14>

<item 15>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
I mean, have I even mentioned that they call themselves “SEOs”? Really. They say things like, “As the SEO, my job is to get more traffic.” This title feels
thirsty to be seen as similar to a CEO, to be taken seriously. And compared to the rest of the tech world, SEO has always lacked a certain glamor or a certain
messiah complex. Case in point: while many of the tech CEOs claiming to save the world these days live in Miami, the alligator party was an hour up the coast in
Fort Lauderdale.

“The SEO people are just trying to make money,” said Peter Kent, the author of several dozen explanatory tech books, including SEO for Dummies and Bitcoin for
Dummies. “The cryptocurrency people are trying to make money, but they’re also trying to overthrow, you know, the existing system.”

Kent has done his fair share of SEO jobs but also has something of an outsider’s perspective.
</item 15>

<item 16>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Public policies attempt to fulfill one or more of the following objectives: efficiency, equity, and stability. Efficiency and stability are necessary conditions
for sustainable growth while equity helps make development inclusive.

This is the second article in our series following the 2nd SGV Tax Symposium, which focused on how a sustainable and effective tax ecosystem can advance the
sustainability agenda for both the public and private sectors.

In this article, we will discuss the Philippine strategy for sustainable development.

GROWTH THROUGH TRADE AND INVESTMENT

The Philippine Development Plan (PDP) 2023-2028 aims for a deep economic and social transformation to reinvigorate job creation and accelerate poverty reduction
by steering the economy back to a high-growth path. In this regard, two main performance indicators are identified and will be monitored by the National
Government.

The first is for the country to graduate into upper middle-income class status within the term of the current administration.
</item 16>

<item 17>
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
</item 17>

<item 18>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
The stock would have been easily missed if PE was the metric used to spot the performers.

Hence EV-based metrics are options to consider as well when equity-based valuation multiples do not give a clear picture. Where even EBITDA multiples are not
clear, as was the case with many new-age company IPOs that were reporting losses even at the EBITDA level, investors have no option but to go further up the P&L
to metrics like EV/revenue, or even outside of P&L to use metrics like EV/subscriber.

As we mentioned earlier, relative valuation represents a number backed by a credible and well-validated story. Hence, you can use non-P&L multiples. So when you
compare companies and invest in one based on EV/Subscriber, there must be a credible story backing future monetisation of the subscriber in a profitable manner.
But do bear in mind, the risks are higher and so may be the rewards or pain as the case may be. When the risks are higher, you must apply a higher margin of
safety before choosing the stock.
</item 18>

<item 19>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Chief among them? Location: “Hevea trees are primarily grown in Southeast Asia,” he says, “while guayule is grown in the Southwestern US.”

This is also a climate resiliency play. Guayule provides an alternate drought-resistant source for natural rubber at a time when global temperature patterns are
behaving like the stock market: highly unpredictable but trending upward.

Virtually every tire manufacturer is making at least token efforts to increase the usage of sustainable materials like guayule. Yokohama is using motorsports to
develop its Advan A005 tire, which debuted at last year’s Pikes Peak International Hill Climb and contains 33 percent sustainable materials, including
biomass-derived rubber. Michelin, meanwhile, shod Porsche’s prototype electric GT4 e-Performance racer with tires said to be 53 percent renewable, like strap
steel and lemon peels.

That all may sound like marketing plays — and it partly is — but motorsports have historically proven an ideal development opportunity in the automotive
industry.
</item 19>

<item 20>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Some people consider EEAT part of what’s making results better than ever. Others see it as a form of censorship, disproportionately affecting right-wing
perspectives. Not every search query takes EEAT into account; Google has described heightened concern over sites that could impact safety, happiness, and the
ability to be an informed citizen. But the point that really hit me was that for certain kinds of information, Google had undone one of the fundamental elements
of what had made its results so appealing from the start. Now, instead of wild-west crowdsourcing, search was often reinforcing institutional authority.

You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at

This felt complicated at best. When it comes to health and wellness, for example, quackery is often in the eye of the beholder. Everyone knows someone who has
struggled with the limits of Western medicine. So much of the original draw of the internet was the opportunity for outlier voices to be heard alongside
established experts and elites.
</item 20>

<item 21>
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
</item 21>

<item 22>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
“The adhesion between the polyanionic and polycationic hydrogel components is remarkably robust, effectively mitigating the swelling of the hydrogel even in the
absence of a cellulose paper scaffold,” Fan explained. “The unexpectedly strong iodine adsorption capability of the positively charged hydrogel has led to a
striking and rapid color transformation, exceeding our initial expectations.

Despite promising results, Fan acknowledged there is much to do before the battery can be brought to the mass market.

“We have to note that in terms of mass or volume energy density, so far, the ZIB is still far from the commercial lithium-ion batteries,” Fan conceded. “The
hydrogel electrolytes need to be further improved in terms of ion conductivity and stability. In addition, hydrogel electrolytes should be compatible with other
battery components, including electrodes and separators.”

The researcher also pointed out that large-scale and cost-effective production methods for hydrogel electrolytes are a prerequisite before the aqueous zinc
iodine batteries is ready for widescale commercial use.
</item 22>

<item 23>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
In other words: less harmful rubber, more grip, better wear, and better range or fuel economy.

The molecular menace

Firestone’s IndyCar tires.

Drive long enough, and the well-defined tread patterns of even the most efficient tires will fade, slowly balding like a stressed-out environmental scientist’s
pate. It’s easy to imagine that the lost rubber simply disappears, but the truth is more terrifying.

Tires break down into tire and road wear particles, or TRWP. This black dust accumulates on roads mostly — but not permanently. When it rains, it is carried
away.

A recent Pew Charitable Trusts report found that 78 percent of ocean microplastics are synthetic tire rubber, which makes efforts to cut down on microfiber
pollution from our laundry seem a bit quaint.
</item 23>

<item 24>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Part of / We only get one planet

Tires are the most important safety component on a car, yet there’s increasing research that they’re doing a good deal of harm, too. Slowly, surely, one mote of
black dust at a time, tires are leaving their mark on the environment.

According to researchers at Imperial College London, our cars and trucks emit 6 million tons of tire wear particles every year, transportation detritus that
goes into the air, the land, and the water. That’s a massive figure that ignores the nearly 250 million tires that are disposed of annually in North America,
roughly half of which are burned.

It’s a depressing state of affairs, but there is room for optimism. Every major tire manufacturer globally is raising its game — not only in terms of
performance in the face of bigger, heavier electric cars but also in terms of environmental impact. Renewable materials are finding their way into modern tires,
and there’s even greener rubber on the horizon, too.
</item 24>

<item 25>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
This definition captures a macro concept of digital twins, but our report and previous article present a three-dimensional, cuboid model to help classify
digital twins so apple-to-apple comparisons can be made. Each axis of the cuboid represents one dimension of a digital twin:

1. Life cycle phase: The X-axis represents the six life cycle phases a digital twin is used for, from design to decommissioning.

2. Hierarchical levels: The Y-axis represents a digital twin’s five hierarchical levels, from information to multi-system.

3. Use/purpose of implementation: The Z-axis represents the seven most common uses for digital twins, such as simulation and prediction.

Based on this model, there are 210 potential combinations (5 x 6 x 7 = 210); however, our research indicates that many digital twin initiatives are tailored to
multiple combinations.
</item 25>

<item 26>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
That isn’t bad just for the salmon and the Pacific Northwest’s fishing industry. 6PPD-quinone is in us, too, and it only comes from tires. 6PPD helps tires
resist environmental damage, thus lasting longer between replacements. When 6PPD reacts with ozone, it becomes 6PPD-quinone.

“The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the
environment,” Goodyear SVP of global operations and CTO Chris Helsel says. That’s hardly a mea culpa, but the tire industry genuinely has been an active partner
in studying the problem.

Goodyear and other members of the US Tire Manufacturers Association actually encouraged 6PPD-targeting regulation enacted by the California Environmental
Protection Agency and have provided free tire compound samples to researchers.
</item 26>

<item 27>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Certainly most of the R&D budget inside of Toyota is spent figuring out how to make the next car or maybe the car five years from now. But there’s this notion —
and I’m sure Hyundai shares this — that we’re at the once-in-a-century time for transformation in the car industry. Battery electric vehicles are much simpler
to make than cars have been in the past — no engine, no transmission, etc. So, we’re going to have to compete in a much fiercer way in cars. But also, can we
use the skills, the dreams and the hopes of the company to go beyond cars? TRI’s job is to actually think about [what’s] next. What’s next after cars, as well
as some fancy stuff on cars.”

For TRI, much of the “what’s next” focuses on supporting aging populations. The organization invests a good portion of its resources to building out technology
designed to help older people live more independently. That’s at the root of the research we shared from the institute last week.
</item 27>

<item 28>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins,” says Nick Molden, CEO of Emissions Analytics, an independent group focused on testing and data about the overall environmental impacts of
transportation.

“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins.”

One such toxin is 6PPD-quinone. A 2020 study from the University of Washington’s Center for Urban Waters found that a dramatic decline in coho salmon
populations was the result of the toxin. The salmon have now been placed on some West Coast endangered lists.

“Shortly after birth, they were essentially going mad and dying,” Molden says. “The researchers finally linked it to this compound in tires because it was
happening after rainfall. The rainfall was washing all the accumulated tire material off the road into the harbors and into the shore.”
</item 28>

<item 29>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
A number of works outline the contours and dynamics of the current media ecology and what it does — Siva Vaidhyanathan’s Antisocial Media, Safiya U. Noble’s
Algorithms of Oppression, Geert Lovink’s Sad by Design, Shoshana Zuboff’s Surveillance Capitalism, Richard Seymour’s The Twittering Machine, Tim Hwang’s
Subprime Attention Crisis, Tressie McMillan Cottom’s writing on how to understand the social relations of Internet technologies through racial capitalism, and
many more. At the same time, there’s reasonable counter-discussion about how many of our problems can really be laid at the feet of social media. The research
on the impacts of social media on political dysfunction, mental health, and society writ large does not paint a neat portrait.
</item 29>

<item 30>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
One of the major drawbacks of these lithium-ion batteries is the inherent risk of burning or even explosion as the result of thermal runaway, with overheated
lithium-ion batteries deemed responsible for a number of fires in the past.

A possibility with zinc

One potential substitute for lithium-ion batteries are rechargeable zinc iodine batteries , which have the advantage of being safer, costing less, and being
more friendly to the environment than lithium alternatives.

However, zinc iodine batteries are not perfect. Composed of an negative iodine electrode and a positive zinc electrode , these batteries suffer from issues like
lower energy output compared to commercial non-aqueous lithium batteries and the growth of zinc dendrites on the anodes — projections of material that build on
electrodes and short batteries when they cross from one electrode to another — as well as the corrosion of the zinc anode that result in reduced efficiency and
even short-circuiting.
</item 30>

<item 31>
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
</item 31>

<item 32>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This has the effect of mitigating the diffusion back and forth of polysulfide between anode and cathode, the so-called “shuttle effect”, which can cause poor
cycling stability and severe anode corrosion in batteries. The use of a hydrogel electrolyte also has the effect of accelerating the iodine conversion reaction
on the cathode side.

Fan explained that the anode and cathode of the hydrogel have different functional groups that make it both zincophilic (with an affinity for zinc ) and
iodophilic ( with an affinity for iodine), respectively.

On the anode side of the battery, this has the effect of repelling dissolved polyiodides — negatively charged ions composed entirely of iodine atoms — allowing
the zinc to be deposited uniformly, protecting the anode from iodine-caused corrosion.
</item 32>

<item 33>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.

DTC digital twin interoperability framework

The Digital Twin Consortium (DTC), a collaborative partnership of industry, academia, and government experts, released the Digital Twin System Interoperability
Framework to “unify a nascent ecosystem of high-value, multi-vendor services that can seamlessly ‘plug into’ a multi-dimensional, interoperable system of
systems.”

DTC and IDTA collaboration

DTC and the Industrial Digital Twin Association (IDTA) announced a liaison agreement to collaborate on standardization requirements, enabling interoperability
through discussions, aligning work in horizontal domains, and collaborating on open-source projects, contributions, and reference implementations.

DTC and OPC Foundation collaboration

DTC and the OPC Foundation announced a liaison agreement to “accelerate the development and adoption of digital twin-enabling technologies,” promoting
interoperability standards and processes to advance the use of digital twins in manufacturing across multiple industries.
</item 33>

<item 34>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Noel P. Rabaja the strategy and transactions (SaT) service leader of SGV & Co.
</item 34>

<item 35>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Truly sustainable tires are becoming a reality, but will it be enough?

Renewable rubber

Bridgestone’s “Wireless-In-Wheel” technology.

If you watched any IndyCar road course race this year, basically any of the half-dozen events where the drivers had to turn right as well as left, you might
have noticed some green flashes on those cars’ tires. It wasn’t just a visual flair to an otherwise visually dull part of the equation.

Those tires contained guayule, part of an agrarian prototype effort from Bridgestone, the supplier of spec tires for that series. It’s a high-profile
demonstration of a new domestically sourced, renewable source of rubber that augments other natural sources like that from the hevea tree and even the Russian
dandelion.

Truly sustainable tires are becoming a reality, but will it be enough?

Bridgestone’s Bill Niaura, executive director of sustainable innovation and circular economy, says guayule is a drop-in replacement with many potential
advantages. Chief among them?
</item 35>

<item 36>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market snapshot

Data from our Digital Twin Market Report 2023–2027 indicates that 29% of global manufacturing companies have either fully or partially implemented their digital
twin strategies, marking a noticeable increase from 20% in 2020. Moreover, the proportion of these companies not contemplating the implementation of digital
twins appears to have reduced to 9% in 2023, down from 33.6% in 2020.

Supporting this trend, an analysis of job postings on SimplyHired suggests a growing demand for digital twin expertise. Postings mentioning “digital twin”
increased by 11% in October 2023 compared to just two years prior, while openings for most other tech topics declined in the same timeframe. Notably, among the
60 tech-related skill sets we tracked, digital twin-related skills experienced one of the most significant growths.
</item 36>

<item 37>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
What comes to mind? If you recombine the familiar characters from these stories, perhaps it looks like climate activists using the capabilities of the Internet
to further both networked protest and energy democracy. In particular, advocacy for a version of “energy democracy” that looks like wind, water, and solar;
decentralized systems; and local community control of energy.

In this essay, I would like to suggest that this is not actually where the three forces of rising authoritarianism x climate change x tech platforms domination
leads. Rather, the political economy of online media has boxed us into a social landscape wherein both the political consensus and the infrastructure we need
for the energy transition is impossible to build. The current configuration of the Internet is a key obstacle to climate action.

The possibilities of climate action exist within a media ecosystem that has monetized our attention and that profits from our hate and division. Algorithms that
reap advertising profits from maximizing time-on-site have figured out that what keeps us clicking is anger.
</item 37>

<item 38>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 38>

<item 39>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.
</item 39>

<item 40>
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
</item 40>

<item 41>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
The polyiodides are immobilized at the cathode side of the battery, and reactions in which iodine is both stripped of electrons (oxidization) and also granted
electrons (reduction) — so-called “redox reactions” — are sped up thanks to the synchronized rapid transport of ions and electrons.

The solid nature of the hydrogel electrolyte can also limit the growth of dendrites thanks to its mechanical strength, which acts as a physical barrier not
present in liquid electrolytes.

Tentative first steps

Testing their aqueous hydrogel-modified zinc iodine batteries battery, the team found that the decay rate was ultralow, with the energy storage capacity of the
battery retained to an outstanding level over the course of 18,000 discharge and recharge cycles. This outstrips the stability of currently available zinc
iodine batteries, and their test battery also delivered some surprising results.
</item 41>

<item 42>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 4: Initiatives promoting interoperability of digital twins across systems from different vendors

Agreement on integration standards among the various digital twin technology providers is crucial for building cross-system digital twins. With standards,
manufacturers can offer services that can be applied to unique situations with other digital twin technology and software.

Recognizing this, countries and industry organizations have taken several steps to address this need:

Plattform Industrie 4.0 and CESMII

Plattform Industrie 4.0 and the Clean Energy Smart Manufacturing Innovation Institute (CESMII) partnered to establish standards for manufacturers in Germany and
the United States to better enable smart, sustainable competition. The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.
</item 42>

<item 43>
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
</item 43>

<item 44>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Example: In large motor applications, such as room-sized motors that pump high volumes of gas, oil, or other chemicals, operators need to monitor the motors’
temperatures—especially if the motors are high-powered and repeatedly started. If a motor is too hot during a restart, the components could cause serious
damage. Unfortunately, using a direct, physical sensor within a motor is often impractical, and operators must work from an assumptive time it takes for the
motor to cool (with an additional safety buffer time). This means motors can be down longer than necessary, impacting efficiency and revenue.

To address this issue, Siemens developed a prototype virtual sensor based on a digital twin. It offers a simulation of how a physical sensor would operate if it
were possible to install it within a motor. Using AR headsets, operators can see a simulation of the motor and its interior with a demonstrator superimposed
over it, and they can see the motor’s temperature.
</item 44>

<item 45>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation specialists

In the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure Digital Twins,
that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver end-to-end digital
twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.

Hyperscalers and OT companies

OT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with hyperscalers, OT companies
can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and computation power.

Example: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.
</item 45>

<item 46>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
because it offers the best possible place to make their vision happen.” Unless his long-term vision was only meant to last until the end of the summer, for
entrepreneurs, innovators and businesses to thrive and unlock the economic potential that comes from creating new industries, we need a consistent approach from
the government.

We have enormous fiscal potential. Within the U.K. climate tech community, we are working to create high-paying jobs and value for investors across just about
every asset class. And collectively — hell, individually — our solutions could genuinely change the world.

Of course, this is the common cause we should be united around. Our company is trying to move industry away from a reliance on fossil fuels to make chemicals,
among other things. But the impact of these political spasms and contradictions in our climate commitments has on, say, an EV battery business — which has just
seen the market demand for its output slide down the road by five years — isn’t hard to appreciate.
</item 46>

<item 47>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Molden believes that what’s needed is better understanding. To that end, he’s recently created the Tyre Emissions Research Consortium, a LinkedIn group with
about 1,000 members. The intent is to break down the silos that individual researchers were working in.

Research into 6PPD replacements is ongoing, and the US Tire Manufacturers Association is also touting research into various ways to mitigate and reduce tire
debris pollution.

Given all that research is happening at an industry level, it’s easy to feel a little helpless about the role we’re all playing in contributing this pollution
to the environment when we’re just trying to get to work. Still, there are things that we can do to help minimize the environmental impact in the interim.

Molden suggests a few things: drive more slowly, drive more smoothly, and choose the right tires. According to Molden, the tires that your car was outfitted
with when new will provide the best wear characteristics.
</item 47>

<item 48>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
It will also promote a green and blue economy coupled with improved governance to guarantee long-term climate and disaster resilience.

BUILDING RESILIENCE THROUGH SUSTAINABLE DEVELOPMENT

Coming off from the pandemic and with the current global economic climate, it is opportune for the government to proactively drive for actionable policies and
programs that focus on building the resiliency of the economy through sustainable development with greater emphasis given to addressing climate change.

Navigating external and domestic economic headwinds will not be an easy feat for the administration, but the private sector will be an important catalyst for
sustainable development. With the private sector sharing industry knowledge, resources and potentially even leading certain socio-economic programs and projects
of the National Government, it is to be hoped that AmBisyon Natin 2040 of long and healthy lives for Filipinos that are strongly rooted, comfortable, and secure
will be achieved.

This article is for general information only and is not a substitute for professional advice where the facts and circumstances warrant. The views and opinions
expressed above are those of the author and do not necessarily represent the views of SGV & Co.
</item 48>

<item 49>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Even worse, the system is addictive, with notifications delivering hits of dopamine in a part of what historian and addiction expert David Courtwright calls
“limbic capitalism.” Society has more or less sleepwalked into this outrage-industrial complex without having a real analytic framework for understanding it.
The tech platforms and some research groups or think tanks offer up “misinformation” or “disinformation” as the framework, which present the problem as if the
problem is bad content poisoning the well, rather than the structure itself being rotten. As Evgeny Morozov has quipped, “Post-truth is to digital capitalism
what pollution is to fossil capitalism — a by-product of operations.”
</item 49>

<item 50>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
The bad news is that we’re still going through more tires than ever. Harrigle says we scrap 250 million tires every year in North America. Again, about half are
burned for fuel.

But there are other solutions. Recycling is one, but reuse is better, a process called retreading. Retreading itself is nothing new, but while it’s generally
fallen out of favor for consumer tires, on the industrial side of things, it can be a huge cost-saver — and environment-saver, too.

Jason Roanhouse, VP of North American Bandag operations at Bridgestone Mobility Solutions, says that the company’s retread business has saved 300 million tires
from being thrown out, worth roughly 4 billion gallons of oil.

Future tires

While efforts to find sustainable replacements for tire manufacturing compounds continue, as does research into what today’s compounds do to our health, many
manufacturers are trying to change the equation altogether.

A near-future example is the Michelin Uptis, a non-pneumatic tire.
</item 50>
=== END ITEM id=eb0513802241f702 ===

=== ITEM id=83a8cf0708193330 ===
Gold sentences and facts:
Sentence 0: The guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.
  Fact 0.0: The guidance lays out ways to use AI responsibly in the courts.
  Fact 0.1: The guidance says courts should be aware of potential bias when using AI.
  Fact 0.2: The guidance says courts should uphold privacy when using AI.
Sentence 1: They promised “invest[ment] in the AI-powered technology” that, amongst other things, would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
  Fact 1.0: The party referred to as 'they' promised investment in the AI-powered technology.
  Fact 1.1: The AI-powered technology would, among other things, help YouTube protect its community of viewers, creators, artists and songwriters.
  Fact 1.2: The AI-powered technology would help YouTube protect its community from trademark and copyright abuse.

Retrieved items:
<item 1>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The report also claims Microsoft has set up a “robust user reporting and appeal process to review and respond to user concerns of harmful or misleading
content”.

Over the next six months, the report does not commit Bing Search to any specific additional steps to address risk attached to the use of generative AI —
Microsoft just says it’s keeping a watching brief, writing: “Bing is regularly reviewing and evaluating its policies and practices related to existing and new
Bing features and adjusts and updates policies as needed.”

TikTok

In its report, TikTok focuses on AI-generated content in the context of ensuring the “integrity” of its services — flagging a recent update to its community
guidelines which also saw it modify its synthetic media policy “to address the use of content created or modified by AI technology on our platform”.

“While we welcome the creativity that new AI may unlock, in line with our updated policy, users must proactively disclose when their content is AI-generated or
manipulated but shows realistic scenes,” it also writes.
</item 1>

<item 2>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Who owns the copyright on ChatGPT-created content or media?

The user who requested the input from ChatGPT is the copyright owner.

What lawsuits are there surrounding ChatGPT?

None specifically targeting ChatGPT. But OpenAI is involved in at least one lawsuit that has implications for AI systems trained on publicly available data,
which would touch on ChatGPT.

Are there issues regarding plagiarism with ChatGPT?

Yes. Text-generating AI models like ChatGPT have a tendency to regurgitate content from their training data.
</item 2>

<item 3>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
As creative industries grapple with AI’s explosion into every artistic medium at once, separate calls from artists warning the world to take action before it’s
too late are starting to converge. From fake Drake songs to stylized Instagram profile pictures, art conjured with newly sophisticated AI tools is suddenly
ubiquitous — and so are conversations about how to rein in the technology before it does irrevocable harm to creative communities.

This week, digital rights organization Fight for the Future partnered with music industry labor group United Musicians and Allied Workers to launch
#AIdayofaction, a campaign that calls on Congress to block corporations from obtaining copyrights on music and other art made with AI.

The idea is that by preventing industry behemoths like major record labels, for example, from copyrighting music made with the assistance of AI, those companies
will be forced to keep looping humans into the creative process. But those same concerns — and the same potential strategies for pushing back against the
onslaught of AI — exist across creative industries.
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
On iPad, ChatGPT now runs in full-screen mode, optimized for the tablet’s interface.

May 2023

Texas judge orders all AI-generated content must be declared and checked

The Texas federal judge has added a requirement that any attorney appearing in his court must attest that “no portion of the filing was drafted by generative
artificial intelligence,” or if it was, that it was checked “by a human being.”

ChatGPT app expanded to more than 30 countries

The list of new countries includes Algeria, Argentina, Azerbaijan, Bolivia, Brazil, Canada, Chile, Costa Rica, Ecuador, Estonia, Ghana, India, Iraq, Israel,
Japan, Jordan, Kazakhstan, Kuwait, Lebanon, Lithuania, Mauritania, Mauritius, Mexico, Morocco, Namibia, Nauru, Oman, Pakistan, Peru, Poland, Qatar, Slovenia,
Tunisia and the United Arab Emirates.
</item 4>

<item 5>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 5>

<item 6>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
GPAIs that fall in the AI Act’s high risk tier will face ex ante-style regulatory requirements to assess and mitigate systemic risks — meaning they must
proactively test model outputs to shrink risks of actual (or “reasonably foreseeable”) negative effects on public health, safety, public security, fundamental
rights, or for society as a whole.

While “low tier” GPAIs will only face lighter transparency requirements, including obligations to apply watermarking to generative AI outputs.

The watermarking requirement for GPAIs falls in an article that was in the original Commission version of the risk-based framework, presented all the way back
in April 2021, which focused on transparency requirements for technologies such as AI chatbots and deepfakes — but which will now also apply generally to
general purpose AI systems.

“There is an obligation to try to watermark [generative AI-produced] text on the basis of the latest state of the art technology that is available,” the
Commission official said, fleshing out details of the agreed watermarking obligations.
</item 6>

<item 7>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the authors of those works, the
pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to favor a holding of fair use , not a reason to
reject it.”

the output of that model might … Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the
authors of those works, the pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to , not a reason to
reject it.” And then the kicker: “Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would therefore undermine the
purpose of copyright and foreclose the many opportunities that come with this technology.”
</item 7>

<item 8>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Marketing claims made for machine learning systems that companies adopted in order to not fall behind will receive their quarterly and yearly reviews… and it’s
very likely they will be found wanting.

Expect a considerable customer withdrawal from AI tools as the benefits fail to justify the costs and risks. On the far end of this spectrum, we are likely to
see lawsuits and regulatory action with AI service providers that failed to back up their claims.

While capabilities will continue to grow and advance, 2023’s products will not all survive by a long shot, and there will be a round of consolidation as the
wobblier riders of the wave fall and are consumed.

Apple jumps in

Apple has an established pattern of waiting, watching and learning from other companies’ failures, then blowing in with a refined and polished take that puts
others to shame. The timing is right for Apple to do this in AI, not just because if it waits too long its competition may eat up the market, but because the
tech is ripe for their kind of improvement.
</item 8>

<item 9>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 9>

<item 10>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
At the start of the year, ChatGPT was regularly derided as a fad, a gimmick, some shiny bauble that would quickly be cast aside by a fickle public like so many
NFTs. Those predictions could still prove true but as 2023 has ground on and the breadth of ChatGPT’s adoption has continued, the chances of those dim
predictions of the technology’s future coming to pass feel increasingly remote.

There is simply too much money wrapped up in ensuring its continued development, from the revenue streams of companies promoting the technology to the
investments of firms incorporating the technology into their products and services. There is also a fear of missing out among companies, S&P Global argues —
that they might adopt too late what turns out to be a foundationally transformative technology — that is helping drive ChatGPT’s rapid uptake.

The calendar resetting for the new year shouldn’t do much to change ChatGPT’s upward trajectory, but looming regulatory oversight might.
</item 10>

<item 11>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.
</item 11>

<item 12>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The list of strictly prohibited use-cases of AI will apply sooner, just six months after the law enters into force — which could, potentially, mean bans on
certain “unacceptable risk” uses of AI, such as social scoring or Clearview AI-style selfie scraping for facial recognition databases, will get up and running
in the second half of 2024, assuming no last minute opposition to the regulation springs up within the Council or Parliament. (For the full list of banned AI
uses, read our earlier post.)
</item 12>

<item 13>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 13>

<item 14>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Confirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should automatically be considered fair
use so that protected works are never compensated for usage and creators have no say in the products or business models that are developed around them and their
work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair use.” Prevent the Cloning of
Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use without the express permission of
the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for name, image, and likeness.”
Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for new works that are created.
</item 14>

<item 15>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 15>

<item 16>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“At the moment, technologies are much better at watermarking videos and audio than watermarking text. But what we ask is the fact that this watermarking takes
place on the basis of state of the art technology — and then we expect, of course, that over time the technology will mature and will be as [good] as possible.”

GPAI model makers must also commit to respecting EU copyright rules, including complying with an existing machine readable opt-out from text and data mining
contained in the EU Copyright Directive — and a carve-out of the Act’s transparency requirements for open source GPAIs does not extend to cutting them loose
from the copyright obligations, with the Commission confirming the Copyright Directive will still apply on open source GPAIs.

As regards the AI Office, which will play a key role in setting risk classification thresholds for GPAIs, the Commission confirmed there’s no budget nor
headcount defined for the expert body as yet.
</item 16>

<item 17>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
And that’s all some people need.

Likely there will be some half-hearted efforts to block generated content from being used in this way, but these posts can’t be taken down fast enough by the
likes of Meta and Google, and the idea that X can (or will) effectively monitor and take down such content is implausible. It’s gonna be a bad time!
</item 17>

<item 18>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT can also show bias in its responses, answering in sexist and racist, overtly Anglocentric ways — or regurgitating portions of the data that it was
trained on.

Lawyers have been sanctioned after using ChatGPT to assist in writing motions, discovering — too late — that ChatGPT invented fake lawsuit citations. And scores
of authors have sued OpenAI over the chatbot regurgitating portions of their work — and not receiving compensation for it.

So what comes next? What might ChatGPT’s second year hold, if not more of the same?

Interestingly — and fortunately — some of the more dire predictions about ChatGPT didn’t come to pass. Some researchers feared the chatbot would be used to
generate disinformation on a massive scale, while others sounded the alarm over ChatGPT’s phishing email-, spam- and malware-generating potential.
</item 18>

<item 19>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
He added that “platforms are quick to point to the loopholes in the law as an excuse to drag their feet or to not take the deepfakes down when requested.”

Kooker commended the No FAKES Act policy drafted by senators earlier this year, which he explained “would create a federal property right in one’s voice or
visual likeness and protect against unauthorized AI-generated replicas”.

2. Sony has ‘roughly 200 active conversations taking place’ with AI startups that “include potential equity investments”

Kooker explained that despite the negative impacts of AI, including unauthorized clones and low-quality AI music, there are still “many positive and optimistic
developments to highlight”.

He highlighted legitimate AI startups and established firms “with music ideas that want to partner with the industry”.

According to Kooker, Sony Music has “roughly 200 active conversations taking place with startups and established players” right now about building new
AI-related products and tools.
</item 19>

<item 20>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 20>

<item 21>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 21>

<item 22>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
A person can also specify their family size, so the text-generating AI can give responses about meals, grocery and vacation planning accordingly.

The FTC is reportedly investigating OpenAI

The FTC is reportedly in at least the exploratory phase of investigation over whether OpenAI’s flagship ChatGPT conversational AI made “false, misleading,
disparaging or harmful” statements about people.

TechCrunch Reporter Devin Coldewey reports:

This kind of investigation doesn’t just appear out of thin air — the FTC doesn’t look around and say “That looks suspicious.” Generally a lawsuit or formal
complaint is brought to their attention and the practices described by it imply that regulations are being ignored. For example, a person may sue a supplement
company because the pills made them sick, and the FTC will launch an investigation on the back of that because there’s evidence the company lied about the side
effects.

OpenAI announced the general availability of GPT-4

Starting July 6, all existing OpenAI developers “with a history of successful payments” can access GPT-4.
</item 22>

<item 23>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“About the time that we started the discussion with this artist, we had begun informal talks with a generative AI start-up company whose business model focused
on working with IP rightsholders the ‘right way’.

“In other words, they respect intellectual property rights and want to work with rights holders in ways that enhance and protect the copyrighted works.”

3. SME has asked Congress to embrace a set of principles around generative AI

Kooker told the lawmakers during his speech on Wednesday that “if copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade
marathon that will change the creative and commercial landscape for music”.

With that in mind, Kooker explained that Sony Music has established a set of principles to guide the company’s decision-making around generative AI.

“If copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade marathon that will change the creative and commercial
landscape for music.”
</item 23>

<item 24>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists;

and jointly announced that the video platform was launching an – a program by which new tools and innovations will be developed at YouTube in close conjunction
with artists and the music biz. To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists; Also in
August, YouTube publicly committed to three principles/pledges behind its development of music-based generative AI tools . These pledges promised “appropriate
protections… for music partners who decide to participate” in said tools. They promised “invest[ment] in the AI-powered technology” that, amongst other things,
would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
</item 24>

<item 25>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.
</item 25>

<item 26>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
But WhatsApp, other encrypted messaging services, and digital rights groups say this scanning would require breaking apps’ encryption systems and invading user
privacy. Whitehead says that Ofcom plans to consult on this next year, leaving its full impact on encrypted messaging uncertain.

“We’re not regulating the technology, we’re regulating the context.”

There’s another technology not emphasized in today’s consultation: artificial intelligence. But that doesn’t mean AI-generated content won’t fall under the
rules. The Online Safety Act attempts to address online harms in a “technology neutral” way, Whitehead says, regardless of how they’ve been created. So
AI-generated CSAM would be in scope by virtue of it being CSAM, and a deepfake used to conduct fraud would be in scope by virtue of the fraud. “We’re not
regulating the technology, we’re regulating the context,” Whitehead says.
</item 26>

<item 27>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
There was also agreement on a “two-tier” system of guardrails to be applied to “general” AI systems, such as the so-called foundational models underpinning the
viral boom in generative AI applications like ChatGPT.

As we reported earlier, the deal reached on foundational models/general purpose AIs (GPAIs) includes some transparency requirements for what co-legislators
referred to as “low tier” AIs — meaning model makers must draw up technical documentation and produce (and publish) detailed summaries about the content used
for training in order to support compliance with EU copyright law. For “high-impact” GPAIs (defined as the cumulative amount of compute used for their training
measured in floating point operations is greater than 10^25) with so-called “systemic risk” there are more stringent obligations.

“If these models meet certain criteria they will have to conduct model evaluations, assess and mitigate systemic risks, conduct adversarial testing, report to
the Commission on serious incidents, ensure cybersecurity and report on their energy efficiency,” the parliament wrote.
</item 27>

<item 28>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 28>

<item 29>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
In a roundtable hosted by the FTC this week, the agency brought together figures from across creative industries — from voice acting and science fiction to
screenwriting, music, illustration and even fashion — to delve into how generative AI is affecting creatives.

“I know that generative AI in particular poses a unique set of opportunities and challenges to creative industries,” FTC chair Lina Khan said. “We’ve already
heard significant concerns about how these technologies could virtually overnight significantly disempower creators and artists who may watch their life’s
creation be appropriated into models over which they have no control.”

In the comments, representatives from myriad creative communities expressed concerns around opt-out requirements that by default train AI models on artists’
original work, and how existing copyright law could be a useful if not comprehensive tool for setting out regulatory guardrails.

In the conversation, a representative with the WGA emphasized that while striking writers obtained their own protections in a newly won agreement, the fight for
artists’ livelihoods “doesn’t stop at the bargaining table.”
</item 29>

<item 30>
Sweeping White House AI executive order takes aim at the technology's toughest challenges (Engadget, 2023-10-30T09:00:08+00:00)
It remains to be seen whether those systems will be able to address the concerns of misbehaving models that SEC head Gary Gensler recently raised.

AI watermarking and cryptographic validation

We're already seeing the normalization of deepfake trickery and AI-empowered disinformation on the campaign trail. So, the White House is taking steps to ensure
that the public can trust the text, audio and video content that it publishes on its official channels. The public must be able to easily validate whether the
content they see is AI-generated or not, argued White House officials on the press call.

Adobe

The Department of Commerce is in charge of the latter effort and is expected to work closely with existing industry advocacy groups like the C2PA and its sister
organization, the CAI, to develop and implement a watermarking system for federal agencies. “We aim to support and facilitate and help standardize that work [by
the C2PA],” administration officials said. “We see ourselves as plugging into that ecosystem.”
</item 30>

<item 31>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 31>

<item 32>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“I therefore urge platforms to be vigilant and provide efficient safeguards for this in the context of elections.”

The EU commissioner noted she’s meeting representatives of ChatGPT maker, OpenAI, later today to discuss the issue.

The AI giant is not a signatory to the bloc’s anti-disinformation Code — as yet — so is likely to be facing pressure to get on board with the effort. (We’ve
reached out to OpenAI with questions about its meeting with the Jourova.)

The commissioner’s remarks today on generative AI follow initial pressure applied to platforms this summer, when she urged signatories to label deepfakes and
other AI-generated content — calling on Code signatories to create a dedicated and separate track to tackle “AI production”, and quipping that machines should
not have free speech.

An incoming pan-EU AI regulation (aka, the EU AI Act) is expected to make user disclosures a legal requirement on makers of generative AI technologies like AI
chatbots. Although the still draft legislation remains the subject of negotiations by EU co-legislators.
</item 32>

<item 33>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Commitment 15 in TikTok’s report signs the platform up to “tak[ing] into consideration transparency obligations and the list of manipulative practices
prohibited under the proposal for Artificial Intelligence Act” — and here it lists being a launch partner of the Partnership on AI’s (PAI) “Responsible
Practices for Synthetic Media” (and contributing to the development of “relevant practices”); and joining “new relevant groups”, such as the Generative AI
working group which started work this month as implemented measures towards this pledge.

In the next six months it says it wants to further strengthen its enforcement of its synthetic media policy — and explore “new products and initiatives to help
enhance our detection and enforcement capabilities” in this area, including in the area of user education.

Meta

Facebook and Instagram parent Meta’s report also includes a recognition that “widespread availability and adoption of generative AI tools may have implications
for how we identify, and address disinformation on our platforms”.
</item 33>

<item 34>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
To help address misinformation, Google has also announced that it will soon be integrating new innovations in watermarking, metadata, and other techniques into
its latest generative models.

“Google also recently joined other leading AI companies to jointly commit to advancing responsible practices in the development of artificial intelligence which
will support efforts by the G7, the OECD, and national governments. Going forward we will continue to report and expand upon Google developed AI tools and are
committed to advance bold and responsible AI, to maximise AI’s benefits and minimise its risks.”

Over the next six months Google’s report states it has no additional measures planned for YouTube. But, with generative image capabilities rolling out
internally over the next year, it commits Google Search to leveraging IPTC Photo Metadata Standard to add metadata tags to images that are generated by Google
AI.

“Creators and publishers will be able to add a similar markup to their own images, so a label can be displayed in Search to indicate the images as AI
generated,” Google’s report further notes.
</item 34>

<item 35>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Italy’s data protection authority has just put out a timely reminder that some countries do have laws that already apply to cutting edge AI: it has ordered
OpenAI to stop processing people’s data locally with immediate effect. The Italian DPA said it’s concerned that the ChatGPT maker is breaching the European
Union’s General Data Protection Regulation (GDPR), and is opening an investigation.

March 2023

1,100+ signatories signed an open letter asking all ‘AI labs to immediately pause for 6 months’

The letter’s signatories include Elon Musk, Steve Wozniak and Tristan Harris of the Center for Humane Technology, among others. The letter calls on “all AI labs
to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4.”

The letter reads:

Contemporary AI systems are now becoming human-competitive at general tasks,[3] and we must ask ourselves: Should we let machines flood our information channels
with propaganda and untruth?
</item 35>

<item 36>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
YouTube says it paid out over USD $6 billion to music rightsholders in the year to end of June 2022, with around $2 billion of that coming just from ads on
user-generated content.)

However, the October US Copyright Office filing from Google has definitely put the cat amongst the pigeons at Sony Music HQ. As we’re sure it has at the other
two major record companies.

“Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology.” Google filing with the US Copyright Office, October 30

Witness Universal Music Group’s own filing with the US Copyright Office on the topic of AI. It couldn’t offer a starker contrast to Google’s assertion that
“prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology”.
</item 36>

<item 37>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
The Council’s PR further stipulates that the higher sanction (7%) would apply for violations of the banned AI applications, while penalties of 1.5% would be
levied for the supply of incorrect information. Additionally, it says sanctions of 3% could be imposed for violations of other AI Act obligations but also notes
that the provisional agreement allows for “more proportionate caps” on administrative fines for SMEs and start-ups in case of infringements. So there looks to
be some scope for AI startups to face smaller penalties for infringements than AI giants may invite.

The deal agreed today also allows for a phased entry into force after the law is adopted — with six months allowed until rules on prohibited use cases kick in;
12 months for transparency and governance requirements; and 24 months for all other requirements. So the full force of the EU’s AI Act may not be felt until
2026.
</item 37>

<item 38>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
And it was an incredible new revenue stream for everyone. AI is that with new super tools.”

Kyncl was referring to YouTube’s Content ID system, which scours uploaded videos for copyrighted content (video and audio), then offers the copyright owner the
option to monetize that video, or to request that it be taken down. The innovative system has essentially allowed YouTube users to upload content without paying
much attention to copyright, while ensuring that copyright owners are paid.

When it comes to the proliferation of AI, “we need to approach it with the same thoughtfulness, and we have to make sure that artists have a choice,” Kyncl
added.

And he made it clear that, in his view, simply rejecting AI and fighting against it is not an option.

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.”
</item 38>

<item 39>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.
</item 39>

<item 40>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
are now allowed to use ChatGPT in legal rulings

The U.K. Judicial Office issued guidance that permits judges to use ChatGPT, along with other AI tools, to write legal rulings and perform court duties. The
guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.

OpenAI makes repeating words “forever” a violation of its terms of service after Google DeepMind test

Following an experiment by Google DeepMind researchers that led ChatGPT to repeat portions of its training data, OpenAI has flagged asking ChatGPT to repeat
specific words “forever” as a violation of its terms of service.

Lawmakers in Brazil enact an ordinance written by ChatGPT

City lawmakers in Brazil enacted a piece of legislation written entirely by ChatGPT without even knowing. Weeks after the bill was passed, Porto Alegre
councilman Ramiro Rosário admitted that he used ChatGPT to write the proposal, and did not tell fellow council members until after the fact.
</item 40>

<item 41>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 41>

<item 42>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
In addition, rights can only be enforced around the training of AI when there are accurate records about what is being copied. Otherwise, the inability to
enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents a thriving ecosystem. This requires
strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to ensure incentives are in place so
that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to clearly distinguish human-created
works from AI-created works. The public should know, when they are listening to music, whether that music was created by a human being or a machine.”

Added Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative AI platforms.
</item 42>

<item 43>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But while a few egregious examples of misuse will be punished, a lack of relevant laws specific to it means that it will necessarily only haphazardly be brought
to court. On that note…

Early adopters take new rules by the horns

Big moves like the EU’s AI Act could change how the industry works, but they tend to be slow to take effect. That’s by design, so companies don’t have to adjust
to new rules overnight, but it also means that we won’t see the effect of these big laws for a good while except among those willing to make changes
preemptively and voluntarily. There will be a lot of “we are beginning the process of…” talk. (Also expect a few quiet lawsuits challenging various parts of
laws.)

To that end we can expect a newly flourishing AI compliance industry as the billions going into the technology prompt matching investments (at a smaller scale,
but still considerable) in making sure the tools and processes meet international and local standards.
</item 43>

<item 44>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
But last month something else happened that, we hear, effectively scuppered any chance that Sony Music would get involved with ‘Dream Track’ at this juncture.

On October 30, Google – the parent of YouTube within Alphabet Inc. – issued a filing with the US Copyright Office that outlined its core position on ‘Artificial
Intelligence and Copyright’.

The contents of this document have caused some alarm at Sony Music HQ.

What was in the Google document?

You may have read on MBW last week that Ed Newton-Rex – a generative AI pioneer, but also a published composer – decided to quit his role at Stability AI over
concerns over the company’s position on “fair use” within its own recent US Copyright Office filing.

‘Fair use’ is, as it sounds, the argument that there are cases where copyrighted material can be used and even replicated that are, well, fair enough.
</item 44>

<item 45>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI promises to defend business customers against copyright claims

Bowing to peer pressure, OpenAI it will pay legal costs incurred by customers who face lawsuits over IP claims against work generated by an OpenAI tool. The
protections seemingly don’t extend to all OpenAI products, like the free and Plus tiers of ChatGPT.

As OpenAI’s multimodal API launches broadly, research shows it’s still flawed

OpenAI announced that GPT-4 with vision will become available alongside the upcoming launch of GPT-4 Turbo API. But some researchers found that the model
remains flawed in several significant and problematic ways.

OpenAI launches API, letting developers build ‘assistants’ into their apps

At its OpenAI DevDay, OpenAI announced the Assistants API to help developers build “agent-like experiences” within their apps. Use cases range from a natural
language-based data analysis app to a coding assistant or even an AI-powered vacation planner.
</item 45>

<item 46>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To be fair to Google, its submission does make note of the importance of US copyright law, including when applied to generative AI, striking “the right balance
between the legitimate interests of rightsholders and the equally legitimate interests of the public and succeeding generations of creators”.

Yet it could clearly be argued that Google’s take on copyright “harvesting” by generative AI models sits in odd contrast to the copyright-related reassurances
that YouTube has been careful to make to the music industry during the pre-development, development, and trial launch of ‘Dream Track’.

A final thought…

MBW’s sources close to Sony Music were keen to point out that the company’s general ongoing relationship with YouTube and YouTube Music is a harmonious one.

(YouTube is, after all, now the second largest commercial partner of the major record companies – with a stated ambition to catch up with Spotify in the years
ahead.
</item 46>

<item 47>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
So it will be worth scrutinizing the text that emerges, likely in January or February.

Additionally, while the full regulation won’t be up and running for a few years the EU will be pushing for GPAIs to abide by codes of practice in the meanwhile
— so AI giants will be under pressure to stick as close to the hard regulations coming down the pipe as possible, via the bloc’s AI Pact.

The EU AI Act itself likely won’t be in full force until some time in 2026 — given the final text must, once compiled (and translated into Member States’
languages), be affirmed by final votes in the parliament and Council, after which there’s a short period before the text of the law is published in the EU’s
Official Journal and another before it comes into force.

EU lawmakers have also agreed a phased approach to the Act’s compliance demands, with 24 months allowed before the high risk rules will apply for GPAIs.
</item 47>

<item 48>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
On that front, the European Union may have gone some – if not all – of the way in making that a reality. Its proposed AI Act – currently the subject of
back-and-forth negotiations between the European Commission and the European Parliament – contains a rule that would require AI developers to disclose whether
their AI had been trained on copyrighted materials.

Transparency in the use of AI is also becoming a rule on social media sites where AI-generated content is likely to appear: Both TikTok and YouTube have now
passed rules requiring AI-generated content to be labeled as such.

3. The NMPA is opposed to an ‘opt-out’ regime for use of copyrighted works in AI

One of the questions the US Copyright Office posed in its call for submissions was “should copyright owners have to affirmatively consent (opt in) to the use of
their works for training [AI], or should they be provided with the means to object (opt out)?”

The NMPA made it clear it’s on the side of “opt in.”
</item 48>

<item 49>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 49>

<item 50>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 50>
=== END ITEM id=83a8cf0708193330 ===
