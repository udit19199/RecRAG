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

This file has 3 items.

=== ITEM id=ab7e4b93ffbfdd20 ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: If the debt continues to rise more than the economy, risks will increase.
  Fact 0.1: If the debt continues to rise more than the economy, the government may crowd out private investment.
  Fact 0.2: The government competes with the private sector for funds to service its debt.
  Fact 0.3: Competing with the private sector for funds to service its debt may cause the government to crowd out private investment.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said we worry about the debt-to-GDP ratio in the Philippines.
  Fact 1.1: Marcos said the Philippines' debt-to-GDP ratio stands at about 63%.
  Fact 1.2: Marcos said that 63% is a little high for the Philippines.
  Fact 1.3: Marcos said that 63% is not ideal.
  Fact 1.4: Marcos told members of the US-ASEAN Business Council who visited Malacanang.
  Fact 1.5: The statement was reported by ABS-CBN on Aug. 9, 2023.

Retrieved items:
<item 1>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The National Government plans to raise more revenue to finance the country’s socio-economic needs, largely through a progressive and simplified tax system, more
efficient and effective tax collection measures through digitalization and, to some extent, from policy measures such as value-added tax on digital service
providers and excise tax on sweetened beverages and junk food.

Achieving fiscal stability presupposes the sustainability of fiscal policy, and fiscal stability helps “crowd in” private investment.

PRICE STABILITY AND INFLATION

In his second State of the Nation Address (SONA), President Ferdinand Marcos, Jr. singled out inflation as the country’s most pressing problem. In the first
nine months of 2023, inflation averaged 6.6% — far above the upper end of the target range of 2-4% set by the BSP. If left unchecked, inflation could undermine
growth. Not only does this increase costs to organizations, but it also sets in motion second-round effects as workers start demanding higher wages,
consequently increasing business costs and discouraging investments.

Graph facts:
- A deficit budget for 2024 -MENTIONS-> BSP
- A deficit budget for 2024 -PUBLISHED_BY-> Business World
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> wages
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> investments

Entities: wages, Business World, BSP, investments, Strategizing for sustainable development, National Government, Ferdinand Marcos, Jr., State of the Nation
Address, fiscal stability, private investment, progressive and simplified tax system, digitalization, value-added tax, digital service providers, excise tax
</item 1>

<item 2>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, Bangko Sentral ng Pilipinas, DBM, budget deficits, foreign and domestic debts, budget surplus policy,
economic growth, privatization, revenue generation, expenditure program, budget surplus of P16 billion, Treasury, National Government
</item 2>

<item 3>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, balanced budget, ABS-CBN, US-ASEAN Business Council, Malacañang, Rachel Lacsa, BSP, deficit budgets,
monetary remedies, borrowings, money supply, fiscal remedies, raising taxes, structural deficits
</item 3>

<item 4>
Graph query row:
article: Strategizing for sustainable development
relationship: MENTIONS
connected_entity: private investment
</item 4>

<item 5>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
During the Congressional budget hearing, Finance Secretary Benjamin Diokno optimistically provided updates on the government’s fiscal and revenue collection
performances and presented sources of funding for the national budget. Tax revenue is projected to increase from P3.5 trillion in 2023 to P6.5 trillion in 2028,
while non-tax revenues are expected to decline from P191 billion in 2023 to P184 billion in 2028, Diokno told the Philippine News Agency (pna.gov.ph, Aug. 10).

Diokno said the Finance department aims to gradually reduce net financing or new debt from P1.42 trillion in 2022 to P1.22 trillion in 2024, which will help
bring the country’s debt-to-GDP ratio to almost below 60% by 2025. And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%
and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, Benjamin Diokno, Congressional budget hearing, Finance department, Philippine News Agency, national budget,
debt-to-GDP ratio, Ferdinand Marcos, Jr., government, Philippines
</item 5>

<item 6>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, deficit budget, Amelia H. C. Ylagan, University of the Philippines, traditional politicians, Business
Administration, Philippines, debt
</item 6>

<item 7>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For this, the economy must grow its per capita income above the threshold set by the World Bank, which means a gross national income (GNI) per capita of at
least $4,466. In 2022, the Philippines achieved a GNI per capita of $3,950.

The second is to lower the poverty level from 18.1% in 2021 to single digits by 2028 — the end of the term of the current administration. Both indicators
require high growth rates. For the next year through 2028, the government pencils the growth rate between 6.5% and 8%.

Growth is expected to be investment-led with the implementation of structural changes such as the Corporate Recovery and Tax Incentives for Enterprises (CREATE)
law, which lowered the corporate income tax rate, and the amendments to the Public Service Act (PSA), Foreign Investment Act (FIA), and Retail Trade
Liberalization Act (RTLA), which further liberalized the economy. The new legislation is expected to attract more local and foreign investment, especially in
the liberalized sectors.

Graph facts:
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -MENTIONS-> World Bank
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> World Bank
- annual meetings of the International Monetary Fund and the World Bank -MENTIONS-> World Bank
- Peace Studies scholar -WORKS_FOR-> World Bank
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -MENTIONS-> poverty

Entities: World Bank, poverty, economy, Business World, gross national income, Corporate Recovery and Tax Incentives for Enterprises law, Public Service Act,
Foreign Investment Act, Retail Trade Liberalization Act, Strategizing for sustainable development, Philippines, investment
</item 7>

<item 8>
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

Graph facts:
- Strategizing for sustainable development -MENTIONS-> trade
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -PUBLISHED_BY-> Business World
- Strategizing for sustainable development -MENTIONS-> wages
- Strategizing for sustainable development -MENTIONS-> GDP

Entities: Business World, 2nd SGV Tax Symposium, Philippine Development Plan 2023-2028, sustainable development, tax ecosystem, sustainability agenda,
Strategizing for sustainable development, National Government, Philippines
</item 8>

<item 9>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
This plan gradually narrows the deficit to 3% of GDP by 2028, down from 7.3% in 2022, such that the debt-to-GDP ratio is reduced from nearly 61% in 2022 to a
more sustainable level of at most 53% by 2028.

DEBT MANAGEMENT

The National Government’s debt was less than 40% before the pandemic struck. It expended much of its fiscal space combatting the pandemic, incurring debt and
large budget deficits. While the current 61% debt-to-GDP ratio may be manageable for an emerging economy like the Philippines for some time, the country may not
have the fiscal space to respond to another potential domestic or external shock. If the debt continues to rise more than the economy, risks will increase, and
the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.

The planned fiscal consolidation entails harmonizing the revenue needs with the promotion of investment through the structure and administration of the reformed
incentive system.

Graph facts:
- A deficit budget for 2024 -MENTIONS-> debt-to-GDP ratio
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> budget deficits
- A deficit budget for 2024 -PUBLISHED_BY-> Business World
- Ferdinand Marcos, Jr. -SAID-> debt-to-GDP ratio

Entities: Business World, budget deficits, debt-to-GDP ratio, Strategizing for sustainable development, National Government, private sector, fiscal space,
fiscal consolidation, reformed incentive system, Philippines, investment, pandemic
</item 9>

<item 10>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Inflation is partly driven by supply-side issues and the government is allocating more resources to the agriculture sector to boost production. Mr. Marcos also
mentioned that the National Government had distributed 28,000 new tools and machinery to farmers. An additional 600 km of farm-to-market roads were laid down to
support the 14 million hectares of farmland, enhancing farmer access to markets. In addition, he signed Executive Order No. 28 in May, forming the Inter-Agency
Committee on Inflation and Market Outlook, which is tasked to keep inflation within government targets and boost the economy.

ENHANCING CLIMATE RESILIENCE

While price stability and fiscal sustainability are important macroeconomic issues, environmental sustainability is increasingly gaining importance. Climate
change uncertainties and challenges need to be managed and both the Philippine government and the private sector are hard-pressed to deliver their commitments
to addressing them. Despite climate change risks, the Philippines has the opportunity to position itself as a prime destination of foreign investments against
climate change or environmental, social, and governance (ESG) investments.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> economy
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> economy
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> economy
- There are no sacred numbers -MENTIONS-> economy
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> economy

Entities: economy, Business World, Strategizing for sustainable development, Mr. Marcos, Philippine government, supply-side issues, agriculture sector, Signing
Executive Order No. 28, Inter-Agency Committee on Inflation and Market Outlook, fiscal sustainability, environmental sustainability, private sector, foreign
investments, environmental, social, and governance (ESG) investments, Philippines
</item 10>
=== END ITEM id=ab7e4b93ffbfdd20 ===

=== ITEM id=ebed262ed545eaf0 ===
Gold sentences and facts:
Sentence 0: It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.
  Fact 0.0: It could mean some sort of progress.
  Fact 0.1: It could mean progress if Ratcliffe follows through with his vision to revolutionise the club.
  Fact 0.2: Ratcliffe was born in Manchester.
  Fact 0.3: Ratcliffe has a vision to revolutionise the club.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United put in a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.

Retrieved items:
<item 1>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
That raises doubts over the future of current sporting director John Murtough and chief executive Richard Arnold, although manager Erik ten Hag is believed to
be on a solid footing despite United’s poor start to the season.

Expect Sir Dave Brailsford, the mastermind behind Team GB’s cycling success at the London 2012 Olympics and director of sport for Ratcliffe’s company INEOS, to
have a prominent role in any reshuffle.

Ratcliffe also has ambitious plans to upgrade Old Trafford and turn it into a 90,000-seat stadium.

He wants the deal ratified in time for the January transfer window, however due to FFP concerns, there is no guarantees United will be able to spend big.

Despite the fact he turns 71 a day before the board meeting, it appears Ratcliffe’s plans centre around long-term stability and success rather than a short-term
turnaround in United’s fortunes.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Erik ten Hag
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Erik ten Hag
- Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win -MENTIONS-> Erik ten Hag
- Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign -MENTIONS-> Erik ten Hag
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Erik ten Hag

Entities: Erik ten Hag, Man United, Old Trafford, talkSPORT, Sir Jim Ratcliffe, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim
Ratcliffe as he seeks board vote, Man United takeover, John Murtough, Richard Arnold, Sir Dave Brailsford, Team GB, London 2012 Olympics, INEOS, FFP concerns,
January transfer window
</item 1>

<item 2>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
This was a move mainly influenced by the fact Raine had managed to broker a deal with Todd Boehly to commit to invest £4.5bn to take control of Chelsea, giving
the Glazers grand ideas they could rake in as much as DOUBLE for a global sporting institution like United.

After several rounds of bidding, it quickly became clear the race for United was a straight shootout between Oldham-born Ratcliffe, who had previously offered
£4bn to buy Chelsea, and Qatari Sheikh Jassim bin Al Thani, who also professed to being a lifelong Red Devils fan and set up the ’92 Foundation’ in tribute to
the famous class of ’92 of David Beckham and company.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Man United, talkSPORT, Sir Jim Ratcliffe, Glazers, Raine, Todd Boehly, Sheikh Jassim bin Al Thani, ’92 Foundation, class of ’92, David
Beckham, Bidding race for United, Todd Boehly deal to invest in Chelsea, Oldham, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim
Ratcliffe as he seeks board vote
</item 2>

<item 3>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Jim Ratcliffe’s move to Monaco in 2020 may have been bad news for the British taxpayer, but not for Manchester United fans.

The chemical billionaire looks set to buy a 25 per cent stake in the Premier League giants after fellow bidder, Sheikh Jassim, withdrew his offer.

6 Ratcliffe could still have a big say at United Credit: getty

6 His wisest move could be to land his old neighbour Mitchell

That news hasn’t been too warmly welcomed by United supporters, who would still have to put up with the hugely unpopular Glazer family in control of the club.

It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.

Should Ratcliffe come in, talkSPORT understands that he would take charge of footballing matters, something the Glazers have been accused of having little to no
interest in.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Manchester United, Manchester, Daniel Levy, Jim Ratcliffe, Glazer family, Monaco, footballing matters, Jim Ratcliffe's move to Monaco, Paul
Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United, Paul Mitchell, talkSPORT, Sheikh
Jassim, Tottenham
</item 3>

<item 4>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
The manager’s future

Erik ten Hag’s team are treading water at precisely the wrong moment: as Sir Jim Ratcliffe takes charge of Manchester United’s sporting department before making
sweeping changes in the posts that will directly affect the manager. Richard Arnold has left despite officially staying until the end of the year and into his
role may come Jean-Claude Blanc, the CEO of Ratcliffe’s Ineos Sport, as the man Ten Hag reports to. Ten Hag’s line manager may change too, as Ratcliffe
considers whether a new football director is required, a position held by John Murtough. Ten Hag may have a first clue of his status in Ratcliffe’s new empire
if he is sounded out for his view of a prime factor in his ability to coach and manage the side to success: United’s transfer policy. This leads us to Murtough
…

Will Murtough survive?

The football director has no plans to depart but whether he remains in situ may be a matter for Ratcliffe.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Kasper Schmeichel
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ross Barkley
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Leicester

Entities: Jean-Claude Blanc, Ineos Sport, transfer policy, From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Manchester United,
Erik ten Hag, The Guardian, Sir Jim Ratcliffe, John Murtough, Richard Arnold
</item 4>

<item 5>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Joel and Avram Glazer put the Red Devils up for sale in November 2022 Credit: AFP

5 United supporters are desperate to see the Glazers depart Old Trafford Credit: Twitter

The fact the Glazers will remain in situ has not gone down well with supporters, but this is seen as the first step to Ratcliffe eventually removing the
Americans and taking overall control.

The exact timescale as to when that happens remains vague and how much it will cost Ratcliffe overall could depend on United’s sporting and commercial success
in the coming seasons.

Thursday’s board meeting was pre-arranged and while it is expected that Ratcliffe’s proposal will discussed in detail, sources close to United have indicated to
talkSPORT that it is far from guaranteed it will be put to a vote with several details still to be thrashed out.
</item 5>

<item 6>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Ratcliffe already owns Nice in France and Lausanne in Switzerland Credit: AFP

Initially it was hoped the bidding process would be concluded in time for the summer transfer window, but the longer it dragged on, prompting several
large-scale protests from United fans, the more apparent it became that Joel and Avram were reluctant sellers, as revealed by talkSPORT in March.

Sheikh Jassim’s public statement promising to restore United ‘to former glories’ was also seen as thinly-veiled swipe at the Glazers, which did not help his
negotiating hand.

The Qatari banker’s patience finally snapped and his withdrawal from the bidding process at the end of last week has left the pathway clear for Ratcliffe to
complete what he has always viewed as his dream purchase.

WHAT ARE THE NEXT STEPS FOR RATCLIFFE?

talkSPORT understands Ratcliffe, having originally wanted a much larger up-front stake, has an agreement in principle to buy a 25 per cent share in United.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Manchester United, Switzerland, talkSPORT, Sir Jim Ratcliffe, Joel, Avram, Nice, Lausanne, Sheikh Jassim, the Glazers, United fans, bidding
process, large-scale protests, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote
</item 6>

<item 7>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
In November, it will have been a year since the deeply unpopular Glazer family left Manchester United fans rejoicing by announcing they were open to offers for
England’s most iconic football club.

Eleven months on and the saga that, at times has resembled an X-Factor style judges houses audition, may finally be coming to an end as British petrochemicals
billionaire Sir Jim Ratcliffe closes in on a partial takeover.

5 Sir Jim Ratcliffe has an offer on the table for 25 per cent stake of Manchester United Credit: AFP

Ratcliffe’s £1.3billion offer for a 25 per stake in the club he has supported since childhood will be high on the agenda when United’s board of directors meet
on Thursday, but how did we get to this point, what happens next and what could the future look like at Old Trafford?

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> AFP
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> AFP
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> AFP
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> AFP
- West Indies clinch T20 series after Shai Hope edges hosts past England total -MENTIONS-> AFP

Entities: AFP, Manchester United, England, Glazer family, Old Trafford, talkSPORT, Sir Jim Ratcliffe, Partial takeover of Manchester United, United’s board of
directors meeting, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote
</item 7>

<item 8>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
As and when it does get as far as a vote, Ratcliffe can be confident it will be ratified seeing as half of the 12-strong board are members of the Glazer family
and in the event of a split vote, Joel and Avram as co-chairmen will have the deciding say.

Even once the final deal is agreed, it is likely to take at least a couple of months for Ratcliffe to pass the Premier League’s owners and directors test, which
has been made more stringent since Portsmouth became the first top-flight side to go into administration in 2010.

5 Ratcliffe, one of Britain's wealthiest men, previously tried to buy Chelsea

WHAT DOES THE FUTURE LOOK LIKE UNDER RATCLIFFE’S INEOS GROUP?

Ratcliffe is demanding an executive role and overall control of day-to-day football matters and plans to appoint his own sporting director, with former
Tottenham and Southampton transfer guru Paul Mitchell one of the main names in the frame.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Premier League
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Premier League
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Premier League
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Premier League
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> Premier League

Entities: Premier League, Portsmouth, Southampton, Chelsea, Glazer family, Paul Mitchell, Man United, talkSPORT, Sir Jim Ratcliffe, Joel, Avram, Inside the Man
United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote, INEOS Group, Man United takeover, board vote
</item 8>

<item 9>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey

Entities: From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Michael Regan, Getty Images, Manchester United, Ratcliffe, The
Guardian, Glazers, Ten Hag
</item 9>

<item 10>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Kasper Schmeichel

Entities: Murtough, From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Dan Ashworth, Blanc, Harvard, Fabrice Bocquet, Juventus,
French Tennis, Tour de France, Manchester City, Manchester United, Paul Mitchell, Monaco, Arnold, Paris Saint-Germain
</item 10>
=== END ITEM id=ebed262ed545eaf0 ===

=== ITEM id=74bd25315776eb16 ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings turned things around after a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The defense has also seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams almost seem to know they won't be able to succeed on downfield shots against the defense.
  Fact 1.3: Tight ends had zero completions of 20-plus yards against the defense.

Retrieved items:
<item 1>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Brandon Powell, Kyler Gordon, Dobbs, Bears vs. Vikings NFL 'Monday Night Football' game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, K.J. Osborn, Sporting News, Bears, T.J. Hockenson
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
No one has scored a touchdown yet tonight.

10:40 p.m. — Addison's catch stands, and an Alexander Mattison run has the Vikings at the edge of field goal range. They want more than three points, but it's
early enough that they absolutely could settle for a field goal if it got to that point.

10:37 p.m. — Matt Eberflus throws the challenge flag on a nice grab by Addison to give the Vikings a first down. This one looks like it will be tough to
overturn...

10:34 p.m. — Wow, Jordan Addison would've had a walk-in touchdown, but Dobbs' throw takes him out of bounds. Addison couldn't have been more wide open.

10:33 p.m. — Brandon Powell has been the Vikings' biggest playmaker at two key points tonight. He goes 18 yards here to put Minnesota near midfield.

10:29 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Brandon Powell, Alexander Mattison, Jordan Addison, Matt Eberflus, Dobbs, Bears vs. Vikings game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, Bears
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Saints, Packers, Kevin O'Connell, Fields, playoff chances, wild-card race, division crown, Monday Night Football game, Bears vs. Vikings live
score, updates, highlights from NFL 'Monday Night Football' game, Sporting News, Justin Jefferson, Rams, NFC North, Bears
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Greg Joseph, Vikings, Alexander Mattison, ESPN, Dobbs, Sweat, Brisker, Bears vs. Vikings game, Bears vs. Vikings live score, updates, highlights
from NFL 'Monday Night Football' game, Sporting News, NFLPlus, Bears, ABC
</item 4>

<item 5>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Travis Kelce, CBSSports.com, NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Kmet
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL 'Monday Night Football' game
- Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL 'Monday Night Football' game
- Garoppolo -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- Alex Anzalone -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- Alim McNeill -PARTICIPATED_IN-> NFL 'Monday Night Football' game

Entities: NFL 'Monday Night Football' game, NFL, Ty Chandler, Vikings, ESPN, Jaylon Johnson, T.J. Edwards, Mattison, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, NFLPlus, Bears, ABC
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Fields, Dobbs, Jaylon Johnson, Bears vs. Vikings 'Monday Night Football' game, Addison, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, D.J. Moore, Bears
</item 7>

<item 8>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Josh Oliver, Vikings, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Johnny Mundt, K.J.,
CBSSports.com, Addison, fantasy football, Justin Jefferson, Kirk Cousins
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Minnesota, Josh Dobbs, Bears vs. Vikings game, Adrian Peterson, Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night
Football' game, U.S. Bank Stadium, Sporting News, Arizona, 2024 NFL Draft, Bears, T.J. Hockenson
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That just hung up there. The Bears have been all over Dobbs and the Vikings' offense early on. A taunting penalty after the play hurts Chicago's field position.

Bears 3, Vikings 0

8:44 p.m. FIELD GOAL — The drive fizzles out, and the Bears opt to kick the field goal on 4th & 3. It's good. The Bears are out to an early 3-0 lead, but
they've left plenty of points on the board.

End of first quarter: Bears 0, Vikings 0

8:39 p.m. — What a play by Fields to get the ball to a wide open Kmet! The Vikings brought intense pressure, allowing Kmet to get wide open and go for 24 yards.
The Bears are in business.

8:38 p.m. — 2nd & 22 becomes 2nd & 27, but the Bears make up 17 yards on two plays.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Fields, Monday Night Football game, Dobbs, Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game,
Sporting News, Kmet, Bears
</item 10>
=== END ITEM id=74bd25315776eb16 ===
