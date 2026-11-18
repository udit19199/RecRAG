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

=== ITEM id=2e2576396d835cc2 ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: Risks will increase if the debt continues to rise more than the economy.
  Fact 0.1: The government competes with the private sector for funds to service its debt.
  Fact 0.2: The government may 'crowd out' private investment.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said the Philippines' debt-to-GDP ratio stands at about 63%.
  Fact 1.1: President Ferdinand Marcos, Jr. said that a 63% debt-to-GDP ratio is a little high for the Philippines.
  Fact 1.2: President Ferdinand Marcos, Jr. said that a 63% debt-to-GDP ratio is not ideal.
  Fact 1.3: President Ferdinand Marcos, Jr. made the statement to members of the US-ASEAN Business Council who visited Malacañang.
  Fact 1.4: The statement is cited to ABS-CBN, Aug. 9, 2023.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
Graph query row:
article: A deficit budget for 2024
published_on: 2023-10-08T16:02:10+00:00
passage: A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
“After extensive deliberations and negotiations, the House of Representatives on Wednesday (Sept. 27) approved on third and final reading the P5.768-trillion
national budget for next year, just before Congress adjourns for a month-long break,” the Press and Public Affairs Bureau of Congress announced on its website,
congress.gov.ph. The 310-strong legislative chamber voted 296 against three, with zero abstentions, to approve House Bill (HB) No. 8980 or the 2024 national
budget on third and final reading. Almost unanimously.

House Speaker Ferdinand Martin G. Romualdez noted that the budgeting process was marked by rigorous discussions, particularly concerning confidential and
intelligence funds, which the House meticulously scrutinized to ensure accountability and responsible utilization. He lauded “the timely passage of the 2024
General Appropriations Bill (GAB),” stating that it not only “demonstrated the commitment of the House to upholding its constitutional role but also underscored
its dedication to serving the Filipino people with transparency and accountability.”
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, budget deficit, national budget for 2024, 2023 budget, current administration, 2024 expenditure program,
tax revenues, additional borrowings, government debt, gross domestic product
</item 10>
=== END ITEM id=2e2576396d835cc2 ===

=== ITEM id=ed8846e538a8842d ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III left the game with an oblique injury.
Sentence 1: Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging injuries.
  Fact 1.0: Entering Week 12, Tee Higgins (hamstring) failed to suit up due to nagging injuries.
  Fact 1.1: Entering Week 12, Noah Brown (knee) failed to suit up due to nagging injuries.
  Fact 1.2: Entering Week 12, Treylon Burks (concussion) failed to suit up due to nagging injuries.
  Fact 1.3: Entering Week 12, Kadarius Toney (ankle, hip) failed to suit up due to nagging injuries.

Retrieved items:
<item 1>
Graph query row:
article: Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game
publication_date: 2023-10-01T23:47:00+00:00
publisher: The Sporting News
connected_person: Kadarius Toney
week_13_context: False
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Pete Carroll, oblique strain,
Pick Six Newsletter, Cooper Kupp, ankle injury, Lions, Rams, Seahawks, Kenneth Walker III, D'Onta Foreman, Zach Charbonnet, Bears, Khalil Herbert
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Cooper Kupp, Talanoa Hufanga,
ACL tear, knee injury, Rams, Seahawks, Kenneth Walker III, San Francisco, Kyle Shanahan, Buccaneers, 49ers
</item 3>

<item 4>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chris Olave
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Chris Olave
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> Chris Olave
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints on Week 7 "Thursday Night Football"
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints game

Entities: Chris Olave, New England, Josh Palmer, Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver, Demario Douglas, Rashid
Shaheed, Amari Cooper, Noah Brown, Treylon Burks, Kadarius Toney, Sporting News, Tee Higgins, Geno Smith, Chargers
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- Brock Purdy -MENTIONS-> NFL

Entities: Tommy DeVito, NFL, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Davis
Mills, Cooper Kupp, Texans, Cardinals, C.J. Stroud, Lions, Rams, Geno Smith, Drew Lock, Seahawks, Kenneth Walker III
</item 6>

<item 7>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Jaguars
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jaguars

Entities: Jaguars, Saints, Tyson Campbell, Bengals, CBSSports.com, injuries, Olave, NFL Week 13 injuries: Browns to start Joe Flacco with Dorian
Thompson-Robinson out, Saints' Olave questionable, Browns, Dorian Thompson-Robinson, D'Ante Smith, Jay Tufele, Brenton Strange, Luke Farrell, Tee Higgins
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Raiders

Entities: Raiders, hand injury, game against the Seahawks, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III
among stars to exit, Cooper Kupp, Miami's game against the Raiders, Dolphins, ankle injury, NFL, Rams, Seahawks, Tyreek Hill
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 9>

<item 10>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Packers
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Packers
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Packers

Entities: Packers, Pete Carroll, Sporting News, knee injury, Seahawks, Kenneth Walker III, Aaron Jones, Zach Charbonnet, oblique injury, ACL, Fantasy Football
RB Rankings Week 12: Who to start, best sleepers at running back, 49ers, Fred Warner
</item 10>
=== END ITEM id=ed8846e538a8842d ===

=== ITEM id=d64bb61b056c30a7 ===
Gold sentences and facts:
Sentence 0: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 0.0: Al Saadi said that her frustration with Meta is common among Palestinians.
  Fact 0.1: Al Saadi said that her frustration with Meta is common among Palestinians in occupied territories.
  Fact 0.2: Al Saadi said that her frustration with Meta is common among Palestinians across the diaspora.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores.
  Fact 1.1: Meta wants to shift the burden of monitoring social media usage among teens back to parents.

Retrieved items:
<item 1>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
This law created very specific restrictions around young folks online, limiting data collection and requiring things like parental consent for various actions,
but a lot of tech companies seem to consider it more suggestion than requirement.

You know it is bad news for the company when they request pages and pages of redactions:

This recently happened with Amazon as well, and it turned out they were trying to hide the existence of a price-hiking algorithm that skimmed billions from
consumers. But it’s much worse when you’re redacting COPPA complaints.

“We’re very bullish and confident in our COPPA allegations. Meta is knowingly taking steps that harm children, and lying about it,” AG Bonta told TechCrunch in
an interview. “In the unredacted complaint we see that Meta knows that its social media platforms are used by millions of kids under 13, and they unlawfully
collect their personal info. It shows that common practice where Meta says one thing in its public-facing comments to Congress and other regulators, while
internally it says something else.”

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Congress, Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges, AG Bonta, price-hiking algorithm,
harm to children, lying about it, unlawful collection of children's personal information, COPPA, Amazon
</item 1>

<item 2>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Arkansas
- Northern Lights: Here are the best tips to help you spot the stunning display in the US and abroad -MENTIONS-> Arkansas
- Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ -MENTIONS-> Arkansas
- Helena World Chronicle -LOCATED_IN-> Arkansas
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> Chinese Communist Party

Entities: Arkansas, Chinese Communist Party, Meta, TechCrunch, sexual predators, teen mental health, body comparison issues, lack of teen protection, Meta seeks
legislation that would require parents to approve teens’ app downloads, Davis, TikTok, U.S., Montana
</item 2>

<item 3>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Bejar, app stores, KOSA, online child safety regulations, social networks, parental approval of teens’ app downloads, age
guidelines, Meta seeks legislation that would require parents to approve teens’ app downloads, Congress, Utah, U.S.
</item 3>

<item 4>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Antigone Davis, coalition of 42 states and D.C., lawsuit against Meta over harms to teens and young users, Frances Haugen, harms to
teens and young users, parental approval for teens’ app downloads, parental responsibility for approving teens’ app downloads, Meta seeks legislation that would
require parents to approve teens’ app downloads, Pew Research
</item 4>

<item 5>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> legislation
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> legislation
- Meta -MENTIONS-> Colombia
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp

Entities: legislation, Meta, TechCrunch, app stores, Meta seeks legislation that would require parents to approve teens’ app downloads, U.S. states, Social
media laws, Entertainment ratings, Video games, movies, TV, Davis, smartphones, apps
</item 5>

<item 6>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Mark Zuckerberg
- Want to help your kids achieve their potential? This expert has some advice -MENTIONS-> Mark Zuckerberg
- Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges -MENTIONS-> Mark Zuckerberg
- Every movie and show coming to Netflix in November -MENTIONS-> Mark Zuckerberg

Entities: Mark Zuckerberg, Apple, Meta, TechCrunch, Meta seeks legislation that would require parents to approve teens’ app downloads, Sacha Haworth, Tech
Oversight Project, willful negligence, children’s safety and privacy, Google, big tech
</item 6>

<item 7>
Graph query row:
article: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices
published_on: 2023-10-19T23:43:15+00:00
page: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
passage: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices Instagram users suspect Meta of shadowbanning content about
Palestine. The platform has a history of inherent bias.

Earlier this year, Palestinian-American filmmaker Khitam Jabr posted a handful of Reels about her family’s trip to the West Bank. In the short travel vlogs,
Jabr shared snippets of Palestinian culture, from eating decadent meals to dancing at her niece’s wedding.

“I hadn’t been in a decade, so it’s just like, life abroad,” Jabr told TechCrunch.

“I would get [anti-Palestine] comments,” she recalled. “And I couldn’t respond [to them] or use my account for 24 hours. I wasn’t even posting anything about
the occupation. But fast forward to now and the same shit’s happening.”

In the aftermath of Hamas’ attack on Israelis, Israel’s retaliatory airstrikes and total blockade — cutting access to electricity, water and vital supplies —
have devastated Gaza.
Meta_connection: Meta
</item 7>

<item 8>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
In this way, Meta goes through great lengths to avoid meaningfully complying with COPPA, looking for loopholes to excuse its knowledge of users under the age of
13 and maintain their presence on the Platform.

The other complaints in the lengthy lawsuit have softer edges, such as the argument that use of the platforms contributes to poor body image and that Meta has
failed to take appropriate measures. That’s arguably not as actionable. But the COPPA stuff is far more cut and dry.

“We have evidence that parents are sending notes to them about their kids being on their platform, and they’re not getting any action. I mean, what more should
you need? It shouldn’t even have to get to that point,” Bonta said.

“These social media platforms can do anything they want,” he continued. “They can be operated by a different algorithm, they can have plastic surgery filters or
not have them, they can give you alerts in the middle of the night or during school, or not.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges, Bonta, social media platforms, COPPA
</item 8>

<item 9>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, California, multi-state lawsuit against Meta, Rob Bonta, Children’s Online Privacy Protection Act, Meta turned a blind eye to kids
on its platforms for years, unredacted lawsuit alleges
</item 9>

<item 10>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Arturo Bejar, body image issues among teens, sexual predators, unwanted sexual advances and harassment, Adam Mosseri hauled before
Congress, Meta seeks legislation that would require parents to approve teens’ app downloads, Congress, Instagram, teen safety, Adam Mosseri
</item 10>
=== END ITEM id=d64bb61b056c30a7 ===
