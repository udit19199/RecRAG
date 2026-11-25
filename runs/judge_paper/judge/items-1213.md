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

This file has 7 items.

=== ITEM id=443d09446bba975f ===
Gold sentences and facts:
Sentence 0: Likely with the sole intent of keeping Messi fit in that time off, and the added bonus of raking in significant cash, Inter Miami scheduled a two-match tour of China in early November after their regular season concluded.
  Fact 0.0: Inter Miami scheduled a two-match tour of China in early November.
  Fact 0.1: The China tour was scheduled after Inter Miami's regular season concluded.
  Fact 0.2: The likely sole intent of the China tour was keeping Messi fit during that time off.
  Fact 0.3: The China tour had the added bonus of raking in significant cash.
Sentence 1: Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.
  Fact 1.0: Alvaro Barreal scored a goal in the 78th minute.
  Fact 1.1: Barreal's 78th-minute goal ended any hope Inter Miami had of completing a stunning playoff charge.
  Fact 1.2: Inter Miami's defeat, together with CF Montreal's 4-1 victory over Portland, means Inter Miami are officially out of the postseason running.
Sentence 2: Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all time.
  Fact 2.0: Lionel Messi's arrival at Inter Miami has brought a craze for fans to get a glimpse of him.
  Fact 2.1: Lionel Messi is one of the sport's most talented players of all time.
  Fact 2.2: Lionel Messi is one of the sport's most legendary players of all time.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com
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
</item 3>

<item 4>
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
</item 4>

<item 5>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
To this end, the first goal of the PDP is social transformation which includes, among others, sustained expenditure on social services, mainly education,
health, and social protection.

The first two goals of the PDP help attain the developmental objectives of efficiency and equity, which rest on the pillars of sustainability and resilience.
The third fosters societal resilience: an enabling environment encompassing institutions, macroeconomic stability, and the physical and natural environment.

Institutions are vital to economic acceleration, which is why the government’s steps to enhance the ease of doing business are most welcome. Infrastructure
development also enables an economy to sustain higher levels of growth, which, in turn, catalyzes yet more investment. The Public-Private Partnership (PPP) Act
is up for the signature of President Marcos and, if approved, is expected to further enhance the business atmosphere in mobilizing private resources for
infrastructure development.

The government aims to sustain its infrastructure program at 5-6% of GDP through 2028 amid a six-year medium-term fiscal framework.
</item 5>

<item 6>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.
</item 6>

<item 7>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.
</item 7>

<item 8>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
During the Congressional budget hearing, Finance Secretary Benjamin Diokno optimistically provided updates on the government’s fiscal and revenue collection
performances and presented sources of funding for the national budget. Tax revenue is projected to increase from P3.5 trillion in 2023 to P6.5 trillion in 2028,
while non-tax revenues are expected to decline from P191 billion in 2023 to P184 billion in 2028, Diokno told the Philippine News Agency (pna.gov.ph, Aug. 10).

Diokno said the Finance department aims to gradually reduce net financing or new debt from P1.42 trillion in 2022 to P1.22 trillion in 2024, which will help
bring the country’s debt-to-GDP ratio to almost below 60% by 2025. And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%
and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr.
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=443d09446bba975f ===

=== ITEM id=b92681d684a8ebd5 ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: A year has passed since Altman catapulted ChatGPT to global fame.
  Fact 0.1: Altman has become Silicon Valley's sought-after voice on the promise of artificial intelligence.
  Fact 0.2: Altman has become Silicon Valley's sought-after voice on the potential dangers of artificial intelligence.
  Fact 0.3: Altman's sudden exit was mostly unexplained.
  Fact 0.4: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: The piece covers this week's topics.
  Fact 1.1: Sam Altman is backing a teen's AI startup.
  Fact 1.2: Google held a hardware event.
  Fact 1.3: First impressions of the Pixel 8 Pro are covered.
  Fact 1.4: Flexport drama is covered.
  Fact 1.5: The ongoing FTX fallout is covered.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The board's language is a basis for the prevailing theory.
  Fact 2.1: The way giant tech companies work is a basis for the prevailing theory.
  Fact 2.2: The prevailing theory currently floating around is based on the board's language and how giant tech companies work.

Retrieved items:

=== END ITEM id=b92681d684a8ebd5 ===

=== ITEM id=d20fc2455cce76c6 ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: His career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation about attaining so much success from such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s own messaging. Crocker, in a September meeting with U.S. reporters along with Cone and U.S. Soccer CEO J.T. Batson, said he hoped to have a new head
coach in place by December.

But the initial contact with Hayes was made a couple of months ago, early into the search, with all three top-level executives from U.S. Soccer involved in
those talks. The trio also described the interview process to journalists in that September meeting – a U.S. soccer statement describes it as involving
“psychometrics and abstract reasoning tests, in-depth discussions of strategy, coaching philosophy and the current player pool, as well as evaluation on the
reactions to pressure, culture-building and interactions with players and staff.” USMNT coach Gregg Berhalter went through a similar process, including an
abstract reasoning test, when he was re-hired by Crocker earlier in 2023.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
The specifics of how that will work without Hayes’ involvement remain a mystery. It would be understandable for players to feel like they are still auditioning
for an absent director until May rolls around, while still knowing they must perform at the USWNT standard.

Along these same lines, there is at least the suggestion that the federation could be willing to sacrifice coherent preparation for this upcoming Olympic
tournament to focus more on the longer-term project of the 2027 World Cup. That itself represents a marked change from the expectations and pressure of constant
performance and winning that the team is known for.

Is that a good or a bad thing? Perhaps a little bit of both. The USWNT shouldn’t be ruled out of contention for the Olympics by any stretch, but this past
summer’s World Cup did reveal that the problems facing the team are far more foundational than just poor coaching decisions or the strange midfield chemistry.
Balancing realism with the pressure to win feels like a much more sustainable path forward for the USWNT.
</item 9>

<item 10>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 10>
=== END ITEM id=d20fc2455cce76c6 ===

=== ITEM id=9d96bbe0cc376437 ===
Gold sentences and facts:
Sentence 0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for technologists and designers to be in the process together.
  Fact 0.0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies.
  Fact 0.1: In this innovation cycle, there is a strong need for technologists and designers to be in the process together.
Sentence 1: Stufish partner Ric Lipson (along with artist Es Devlin and producer Brian Eno) collaborated on the U2 project for Sphere.
  Fact 1.0: Ric Lipson is a partner at Stufish.
  Fact 1.1: Ric Lipson collaborated on the U2 project for Sphere.
  Fact 1.2: Es Devlin, an artist, collaborated with Ric Lipson on the U2 project for Sphere.
  Fact 1.3: Brian Eno, a producer, collaborated with Ric Lipson on the U2 project for Sphere.

Retrieved items:

=== END ITEM id=9d96bbe0cc376437 ===

=== ITEM id=aa631af96cf575d6 ===
Gold sentences and facts:
Sentence 0: 10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late.
  Fact 0.0: At 10:59 p.m., the Cowboys are moving the ball.
  Fact 0.1: The Cowboys are moving the ball when it is a bit too late.
Sentence 1: Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati.
  Fact 1.0: Mixon has been forced to step up.
  Fact 1.1: Burrow is not under center.
  Fact 1.2: Mixon is forced to step up without Burrow under center.
  Fact 1.3: Mixon doubles as a runner for Cincinnati.
  Fact 1.4: Mixon doubles as a receiver for Cincinnati.

Retrieved items:

=== END ITEM id=aa631af96cf575d6 ===

=== ITEM id=7798ff2ba2607f70 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 1>

<item 2>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In Alan Wake 2, everyone’s story is happening at once, branching out in endless directions, and we might not know the genre until it’s too late. It’s a
hypertext mystery, an ergodic game that molds itself to an audience raised online, though few computers are in sight. It’s a game for a world suspended in an
endless second act — forever wars, live services, franchises, infinite scrolls — endlessly searching for a conclusion. —JR

2. Baldur’s Gate 3

Developer: Larian Studios

Where to play: PlayStation 5 and Windows PC

Baldur’s Gate 3 hit at the perfect moment. Despite the handful of truly great computer role-playing games that have been released since the early 2000s (Larian
Studios itself made waves with its Divinity: Original Sin series in 2014 and 2017), the genre has largely remained a modern niche. But times have changed:
Dungeons & Dragons has undergone something of a renaissance, thanks to actual-play series like Critical Role and Dimension 20.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 3>

<item 4>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 4>

<item 5>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The GameSir G7 wired controller for Xbox and PC is on sale for $35.99 ($9 off) at Amazon. It’s a straightforward third-party gamepad, though it also has two
programmable rear buttons, clicky face buttons, and a paintable front shell that is attached with magnets.

for Xbox and PC is on sale for $35.99 ($9 off) at Amazon. It’s a straightforward third-party gamepad, though it also has two programmable rear buttons, clicky
face buttons, and a paintable front shell that is attached with magnets. Another nifty GameSir controller on offer is the T4 Kaleid, which is on sale for $33.59
($8 off). This is the first time the transparent wired controller with RGB lighting has gone on sale. It sports drift-free Hall effect sticks and is compatible
with PC, Nintendo Switch, and Android.
</item 5>

<item 6>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At the risk of diminishing its accomplishments, I’m going to make a comparison that should cut to the quick for interested parties: This is the most A Short
Hike-alike that I’ve discovered yet, and I’ve been chasing that singular high for years. So if you have a lazy weekend afternoon and want to spend it playing —
not gaming, but playing, in the joyous, unstructured sense of a day in the park — I can’t recommend this li’l game enough. —Chris Grant

Related Lil Gator Game has the heart of Wind Waker and the adventure of Breath of the Wild

34. Starfield

Developer: Bethesda Game Studios

Where to play: Windows PC and Xbox Series X

Starfield had so much to prove, it’s easy to lose sight of what it accomplished: top-tier world-building, a fantastically customizable ship builder, and
Bethesda’s most engaging combat to date.
</item 6>

<item 7>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
That’s a whole lot of four-player co-op arcade brawling for not a lot of money.

Consoles, controllers, and accessories

Hallmark’s Nintendo Entertainment System Console Ornament is on sale for $16.16 (about $4 off) at Amazon. The tiny NES replica has a little copy of the Super
Mario Bros. in it, and when you press its power button, it lights up and plays the classic Mario theme with sound effects.

Microsoft Xbox Series X $ 400 $ 500 20 % off $ 400 $ 400 $ 500 20 % off The Xbox Series X is Microsoft’s flagship console, serving as its most powerful (and
biggest) option. While the Series S is aimed at smooth 1440p performance and takes a disc-less approach, the $500 Series X is focused on fast 4K gameplay.
</item 7>

<item 8>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 8>

<item 9>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Metroid Dread $ 40 $ 60 33 % off $ 40 $ 40 $ 60 33 % off Metroid Dread is the fifth mainline entry in Nintendo’s storied franchise. The 2D side-scroller is a
direct sequel to Metroid Fusion, a Game Boy Advance title from 2002. Read our review. $40 at Amazon (digital)

Cuphead (Nintendo Switch) $ 28 $ 40 30 % off $ 28 $ 28 $ 40 30 % off Inspired by the animated cartoons of the 1930s, Cuphead is a platformer and bullet-hell
game with a stylish nod to old cartoons. It’s also an unforgiving platformer with a high skill ceiling and little room for error. The Nintendo Switch cartridge
version includes the Delicious Last Course DLC expansion, adding Ms. Chalice as a playable character.
</item 9>

<item 10>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
We’re coming up for air from our coverage of The Game Awards to loop our readers in on some of the best deals happening this weekend. If you’re looking to
experience some of the winners from the 2023 Game Awards for yourself, you can find discounts on a variety of titles on Steam. Some other highlights you might
want to check out include a sitewide sale on Blu-Rays from The Criterion Collection, and the Xbox Series X matching its Black Friday price on Walmart and
Amazon.

As we do every weekend here at Polygon, we’ve rounded up a collection of our favorite deals from the worlds of gaming and entertainment, along with some of the
best-selling products that have made a recent appearance on our site.

The best gaming deals this week

Steam is discounting many contenders from the 2023 Game Awards and winners from the past nine years.
</item 10>

<item 11>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.
</item 11>

<item 12>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 12>

<item 13>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
(Parts will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and
2.5-meter cable that stretches the length of my couch.

One small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra cautious, stick to the original bulky case. But
if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little thumbstick nipples!

All of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam Deck available today,
just that the OLED will come with more stuff.
</item 13>

<item 14>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Email (required) Sign up By submitting your email, you agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy
and Terms of Service apply.
</item 14>

<item 15>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Because while Valve hasn’t upped the horsepower, it has improved practically everything else. The Steam Deck OLED has longer battery life, faster Wi-Fi, and a
stand-alone Bluetooth module. And the screen isn’t just OLED; it’s HDR OLED — the first HDR OLED screen on any gaming portable, with a peak brightness of 1,000
nits for HDR content and 600 nits for SDR (for comparison, the original screen’s peak brightness was 400 nits). And somehow, with all of these additions, the
Steam Deck OLED is lighter.

So, goodbye, old Steam Deck; hello, new Steam Deck.

The Steam Deck OLED delivers on Valve’s original promise

OLED is in the name for a reason.

The OLED panel in the new Steam Deck is lighter, thinner, and more power efficient than its LCD predecessor. It’s also bigger and faster. Though the device has
the same shape as the original Steam Deck, smaller screen bezels allow for a display that’s slightly larger, at 7.4 inches compared to 7 inches.
</item 15>

<item 16>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Again, you can do that by clicking the “Add a Game” option on the bottom left corner in the Steam client.

“This is a very solid, unassuming but beautifully engineered pad, with Nintendo’s typically great ergonomics and switchgear. The D-pad is one of the best out
there, and those lovely, fat, responsive face buttons will give you SNES flashbacks in a good way” —Oli Welsh

Alternatively, if you don’t want to use Steam to play your games, relying on DS4Windows and its bevy of drivers is a free yet slightly advanced way to make
Windows recognize your Switch Pro wireless controller. Of course, that $19.99 8BitDo USB Adapter 2 that I keep referencing can come in handy here, too. After
plugging it into your PC and tethering wirelessly to it, you’re good to go — no fussing with apps required.

The Switch Pro controller is, unsurprisingly, a fantastic Switch controller.
</item 16>

<item 17>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Mike Mahardy

How the Polygon top 50 list works

Over the past few weeks, the Polygon staff voted, championed, debated, and ultimately threw up its hands and marveled at the list of mammoths, curiosities,
puzzle boxes, and black holes that is our top 50 games of 2023. Any video games that were released in 2023, received substantial updates in 2023, or achieved
renewed cultural relevance in 2023 were eligible for this list. Last year, the cutoff for consideration was Nov. 30. (You’ll notice a certain Firaxis Games
joint fairly high up our list.) This year, the cutoff was the same. Should we be thoroughly enamored with Warhammer 40,000: Rogue Trader or Avatar: Frontiers of
Pandora, we’ll make sure they’re considered for next year’s top 50.

Top 50

50. Mr. Sun’s Hatbox

Developer: Kenny Sun

Where to play: Nintendo Switch and Windows PC

Mr.
</item 17>

<item 18>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It performs best in a PC with a PCIe 4.0 motherboard or in a PS5 (with a requisite heatsink). $70 at Amazon (1TB, with heatsink)

The Western Digital WD_Black C50 Storage Expansion Card for Xbox Series X / S consoles is selling for a new low price of $124.99 ($25 off) at Amazon.

Tabletop games

Catan $ 27 $ 49 45 % off $ 27 $ 27 $ 49 45 % off The iconic board game classic that has players strategically competing and trading for resources. $27 at
Amazon$29 at Target

Happy Salmon is selling for just $8.63 ($4 off) at Amazon. The kid-friendly party card game by the Exploding Kittens creators is something you can play to
completion in just a few minutes. It’s a great ice breaker for adults and children alike, getting everyone out of their seats and quickly causing a raucous.
</item 18>

<item 19>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 19>

<item 20>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The PDP Victrix BFG Wireless Controller for PlayStation 5, PS4, and PC in its Call of Duty: Modern Warfare II-branded black-and-gold design is on sale at Amazon
for $169.99 ($30 off). I know this controller looks very Extra (and it is), but it’s a very versatile DualSense alternative if you like FPS and fighting games.
It’s got multi-stage hair triggers, and its sticks and face buttons can be swapped out for an Xbox-style layout or as a six-button fightpad.

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Wireless gaming headsets, Laptops, and PC accessories

Asus ROG Flow Z13 $ 893 $ 1300 31 % off $ 893 $ 893 $ 1300 31 % off The ROG Flow Z13 is a gaming tablet with a detachable keyboard, kickstand, and plenty of
chops for QHD gaming.
</item 20>

<item 21>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
Meanwhile Sophie, who was named Ubisoft's content creator of the year, told the BBC she "just wants the games to win".

"We want everyone to be able to play video games that they want to play, we want them to be accessible for everyone," she said.

"Obviously, we're going to have our opinions, but if the end result is good games, that's fine with us."

This Instagram post cannot be displayed in your browser. Please enable Javascript or try a different browser. View original content on Instagram The BBC is not
responsible for the content of external sites. Skip instagram post by cadaea Allow Instagram content? This article contains content provided by Instagram. We
ask for your permission before anything is loaded, as they may be using cookies and other technologies. You may want to read Meta’s Instagram cookie policy,
external and privacy policy, external before accepting. To view this content choose ‘accept and continue’. Accept and continue The BBC is not responsible for
the content of external sites.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I think the philosophy change would do both of those companies much good.

If you get your way in terms of being able to freely put your own store on Android, do you believe that would only be a game store or would it also be an app
store? We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.

So the Epic Games Store isn’t a games store, right? It’s the store operated by Epic Games. So we have a lot of non-games there already. We have the Brave web
browser, we have a number of software creation tools including Unreal Engine, and there’s more coming, including some other awesome creation tools and
productivity tools. We’ll host any app anybody wants of any sort.

I think the gaming market is something we’re uniquely close to, and so I think we would likely be able to forge closer partnerships and opportunities in gaming,
but we’ll be open to everybody on Android as we are on PC.
</item 22>

<item 23>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 23>

<item 24>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
If you buy something from a Polygon link, Vox Media may earn a commission. See our ethics statement .

Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

A power bank, also commonly referred to as a portable charger, is an essential purchase for anyone who’s on the go with a phone, tablet, or a gaming handheld
like a Nintendo Switch or a Steam Deck. It lets you recharge your precious gadgets one or more times via a USB connection, so you can keep talking, texting, or
gaming throughout the day.

Power banks range in price depending on a few factors: how compact it is, what kind of ports it has, how much battery capacity it contains (typically measured
in mAh), and how quickly it can recharge your devices.
</item 24>

<item 25>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 25>

<item 26>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 26>

<item 27>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router and connection to make use of it) and a dedicated Bluetooth module, which
allows for improved audio quality with wireless headphones, the option to wake the Steam Deck from a controller, and the capability to connect many Bluetooth
controllers for multiplayer. The haptics are noticeably improved, with a wider frequency range. And the touchscreen is markedly better — it’s responsive to the
point where I don’t feel the need to plug in a USB-C keyboard anytime I need to type a few words.

A review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also recommend reading The
Verge’s review.

Valve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for myself. (Parts will once again be sold
through iFixit.)
</item 27>

<item 28>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
$50 at Best Buy

Google Nest Hub (second-gen) $ 60 $ 100 40 % off $ 60 $ 60 $ 100 40 % off The second-gen Nest Hub is Google’s smallest smart display, with a seven-inch
touchscreen that can automatically adjust itself to the lighting in your room. What’s new in the latest version is its ability to automatically track your sleep
patterns with its Soli radar chip. Read our review. $60 at Walmart

The best gaming deals

Sony InZone H9 $ 245 $ 300 18 % off $ 245 $ 245 $ 300 18 % off The H9 is at the top end of Sony’s InZone gaming headsets. It features active noise cancellation,
compatibility with both PCs and the PlayStation 5, and can simultaneously connect to two devices (one over 2.4GHz wireless and one over Bluetooth).
</item 28>

<item 29>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have gotten toasty playing the OLED
model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder, and it clearly works a treat.

The Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a compatible router. I don’t have a 6E
router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was 562mbps, which was about 10 percent
faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3 and adds a dedicated antenna, which
should improve functionality when using Wi-Fi and Bluetooth at the same time.
</item 29>

<item 30>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 30>

<item 31>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 31>

<item 32>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
The Evo Select is our "best value" pick: It's slower than the other two, but it's cheaper, and its losses won't be hugely noticeable in devices that can't take
advantage of the extra performance such as the Switch. Each card comes with a 10-year warranty.

PlayStation DualSense Wireless Controller

Amazon, Walmart, PlayStation Direct and several other retailers have discounted a bunch of PlayStation DualSense controllers ahead of Black Friday, including
many of the colorways that typically go for $75. You can pick up Cosmic Red, Starlight Blue, Nova Pink and other models for $50 each, which is just about a
record-low price. These controllers work with the PS5 as well as PC, and they sport adaptive L2 and R2 triggers, haptic feedback and a built-in mic array.

Xbox Wireless Controller

If you need a spare gamepad for your Xbox, meanwhile, official Xbox Series X/S Wireless Controllers are as low as $40 at Microsoft and Best Buy.
</item 32>

<item 33>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

Finding the best controller for playing PC games is a more personal journey than it sounds. It’s got to be comfortable to use, with every button, stick, and
trigger easily in reach. It’s also important that you’re getting the features you want. For example, those might include easy setup on PC, good battery life,
multi-platform compatibility, and accessibility options, like the ability to remap buttons and adjust stick or trigger sensitivity.

At Polygon, we play games on a lot of consoles and platforms — both new and old. Our staff has a wide variety of preferences and needs. We’ve come together to
share our favorite PC-ready controllers.

Microsoft Xbox controller

On the Polygon team, the Xbox controller got the highest number of endorsements of any controller.
</item 33>

<item 34>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Valve will be selling three models:

$649 - 1 TB OLED (up from 512 GB)

$529 - 512 GB OLED (up from 256 GB)

$399 - 256 GB non-OLED Steam Deck (up from 64 GB)

Plus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for $679 — only in the U.S. and Canada.
Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have I considered ordering one of these
limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?

I suppose that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that
the screen and the battery justify the double dip.
</item 34>

<item 35>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It streams in 4K, supports HDR color, and is compatible with Alexa.

Games and gaming accessories

Mass Effect Legendary Edition $ 12 $ 60 80 % off $ 12 $ 12 $ 60 80 % off A compilation of the first three Mass Effect games repackaged and optimized for 4K with
reduced load times, improved performance, and various gameplay tweaks. $12 at Amazon (Steam code)

Samsung Evo Select microSD card (512GB) $ 25 $ 35 29 % off $ 25 $ 25 $ 35 29 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $25 at Amazon

Staff favorites and miscellaneous tech

ChomChom pet hair remover $ 20 $ 32 38 % off $ 20 $ 20 $ 32 38 % off The ChomChom is a reusable cat and dog hair remover that works great for furniture.
</item 35>

<item 36>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
It’s just impossible to have that much content. So that content needs to come to the device in a very unique architecture where there’s very low latency,
there’s a lot of local 3D simulation, but at the same time, on the cloud, that content needs to load and connect very fast. And I think this architectural
inevitability of a 3D connection platform where we can go everywhere instantly is maybe what you’re thinking about or referring to. I would say, Apple, these
platforms, are very aware of this inevitable architecture for 3D and are actually very big supporters of it.

Andrew Melnizek: Hey. Andrew with The Verge. Last week, we saw some leaked internal emails from Microsoft’s Xbox CEO Phil Spencer, and he talked about the
future of AAA game publishing and development. It’s just really expensive. People have to take bigger bets to succeed, and it’s getting harder and harder.
</item 36>

<item 37>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
All of the buttons, triggers, and sticks feel just as good to use as any modern controller from a console manufacturer.

Like several other 8BitDo controllers, this one can be thoroughly customized with its Ultimate software on mobile devices, or on PC. From there, you can remap
controls, assign buttons to its two rear paddle buttons, tweak the actuation sensitivity of its triggers, and adjust stick settings. The controller allows up to
three custom control schemes, which you can switch between with its middle “star” button.

The Ultimate controller ships with a charging dock, and nesting inside of that dock is a 2.4 GHz transmitter that makes it fast and easy to connect to a PC. The
dock makes for a convenient way to charge the controller, and you can switch between DirectInput and XInput by using a switch on its back, depending on the
requirements of each PC game.

8BitDo makes a pricier version of this controller with Nintendo-style face buttons called the Ultimate Bluetooth. They’re virtually identical, and it can work
via 2.4 GHz with PC, too.
</item 37>

<item 38>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
$245 at Amazon$250 at Best Buy

Seagate Storage Expansion Card (1TB) $ 130 $ 220 41 % off $ 130 $ 130 $ 220 41 % off Proprietary SSD expansion for the Xbox Series X / S consoles. The
plug-and-play drives are designed to be as fast as the Xbox internal SSD and are sold in 512GB, 1TB, and 2TB configurations. $130 at Best Buy

The best TV and streaming device deals

2022 Samsung The Frame TV (55-inch) $ 1000 $ 1500 33 % off $ 1000 $ 1000 $ 1500 33 % off Samsung’s art-inspired 4K TV features a QLED display and an anti-glare
matte display, which provides artwork and photos with qualities that resemble an actual canvas.
</item 38>

<item 39>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Amazon’s fall Prime Day event is underway. For more deals we recommend across all categories, check out our full roundup here.

Amazon Prime Day is one of the best times of the year to save on a wide variety of video games. Titles released in the last few months are getting some decent
discounts (which still beats paying full price), and many older games from the last few years fall to their best bargain-bin prices (which is great for stocking
up on those “Man, I’ve been meaning to get around to that one” games).

Related The best October Prime Day deals available on Day 2

Here, we’ll go over all the best gaming deals we can scrounge up during Prime Big Deal Days across PlayStation 5, PS4, Xbox Series X / S, Nintendo Switch, and
PC. We’re also on the lookout for hardware deals on things like controllers, streaming peripherals, and the consoles themselves.
</item 39>

<item 40>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The immediate reality is a bit of a mess and a little complicated, but this, Valve says, will be the Steam Deck line-up moving forward:

256GB Steam Deck LCD ($399)

512GB Steam Deck OLED ($549)

1TB Steam Deck OLED ($649)

As a result of the changes, the 64GB LCD Steam Deck (previously sold at $399) will drop to $349, and the 512GB LCD (previously $649) will drop to $499. Once
Valve sells through its inventory of both, those lines will be discontinued. Valve says it will continue to support all LCD models with software updates.

So, the long-term lineup starts at the same $399 price, which gets you the old LCD Steam Deck, but with 256GB of storage instead of 64GB. The new step-up model
costs $20 more at $549, but comes with an OLED display and the improvements outlined above, plus double the storage.
</item 40>

<item 41>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 41>

<item 42>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 42>

<item 43>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 43>

<item 44>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 44>

<item 45>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Microsoft owns nearly 40 developers and it bought a chunk of those in the past five years. The Xbox umbrella covers 343 industries, Arkane Studios, Bethesda,
Compulsion Games, Double Fine Productions, id Software, Infinity Ward, Mojang Studios, Ninja Theory, Playground Games, Tango Gameworks and Turn 10, among dozens
more.

Charley Gallay via Getty Images

With these studios at its back, Microsoft is leaning hard into cloud gaming while attempting to build a device-agnostic ecosystem powered by the Xbox brand.
These moves are designed to unlock the mobile market even more, putting Xbox games on all devices, everywhere, all the time.

Still, Sony is bigger than Microsoft by revenue. Though Microsoft is often the face of the game-studio acquisition spree, Sony is the owner of 21 development
teams, including Bungie, Guerrilla Games, Haven Studios, Housemarque, Insomniac Games, Media Molecule, Naughty Dog and Sucker Punch Productions.
</item 45>

<item 46>
6 games to play first on your new Nintendo Switch (Polygon, 2023-12-24T15:00:00+00:00)
Many games are huge. Others are endless. Some are so exciting that they swallow our news feeds whole. That’s where Polygon’s What to Play comes in: We curate
the best, most innovative, and most intriguing games on every platform, so you can spend less time searching, and more time playing.

eCongrats, you are the owner of a Nintendo Switch! It’s a wonderful position to be in, as the console has a robust, delightful library of games. Plus you not
only get to play them on a television screen or monitor; you can also take the console with you wherever you need to go. And it doesn’t hurt that there are tons
of cute Joy-Con colors, a great pro controller, and lots of themed peripherals for fans of all stripes.

The hardest part is where to even start, in terms of sorting through your options. Luckily, we’ve got six recommendations, ranging from modern Nintendo classics
(that you can’t play on any non-Nintendo console) to indie games that are perfect for the Switch.
</item 46>

<item 47>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
One of the company's 67W USB-C chargers is 37 percent off, bringing it to $38 and matching its all-time low. Our top 65W fast wall charger, Anker's 735
GaNPrime, is down to a new low price after a 38 percent discount. And the Anker Nano Power Bank is seeing one of its first discounts since coming on the market
a couple months ago.

Meta Quest 2 VR headset

The Meta Quest 2 VR headset is $50 off and down to $249 at several retailers. That matches the best price we’ve seen. At GameStop and Target, you can get the
device with a $50 gift card. Despite the launch of the impressive Quest 3, we still consider the Quest 2 to be one of the best VR headsets available right now —
precisely because of its more affordable price.
</item 47>

<item 48>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Just try not to make those salmon so happy you end up with bruised forearms (you’ll see what I mean).

is selling for just $8.63 ($4 off) at Amazon. The kid-friendly party card game by the Exploding Kittens creators is something you can play to completion in just
a few minutes. It’s a great ice breaker for adults and children alike, getting everyone out of their seats and quickly causing a raucous. Just try not to make
those salmon so happy you end up with bruised forearms (you’ll see what I mean). Resource management game Splendor is on sale at Amazon for $19.97 ($25 off),
which is one of its lowest prices. It’s a competitive game set in Renaissance Europe, where you compete with up to three other players to build the best jewelry
business — and it’s a modern tabletop classic.

Verge Deals / Sign up for Verge Deals to get deals on products we've tested sent to your inbox daily.
</item 48>

<item 49>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 49>

<item 50>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
You can get this two-pack of 100 W-ready, six-foot-long, USB-C-to-C cables at Amazon for $15.99.

Related The best Nintendo Switch accessories

Best power bank for your Steam Deck

Each Steam Deck console includes a 45 W USB-C wall adapter, so it’s a good idea to find a power bank that can meet that power level when you’re on the go.
Thankfully, some options exist that can do this. One thing to note during your search is that only batteries that support the PD 3.0 spec, including the two
options below, can recharge the Steam Deck at its fastest speed.

Related The best Steam Deck accessories

Best portable charger for your laptop

The number of laptops that can recharge via USB-C is growing, making it possible to use the same charger to refill your laptop as you do your phone or Nintendo
Switch. For instance, Apple’s entire MacBook lineup allows it, as do several popular Windows laptops, like the Dell XPS 13, Razer’s Blade gaming laptop, and
more.
</item 50>

<item 51>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $119.99 Amazon Best Buy

8BitDo SN30 Pro (Special Edition) The 8BitDo SN30 is more than a small controller best suited for tiny hands. It also oozes nostalgic vibes to make you feel a
bit of what mom and dad did while playing the original SNES — albeit, with modern games on PC or the Nintendo Switch. What’s more, its see-through color schemes
are sure to entice just about everyone, regardless of what year you were born.

Price: $44.99+ Amazon Walmart 8BitDo

Nintendo Switch Lite (Isabelle’s Aloha Edition) The Steam Deck may have ushered in a new era of handheld gaming, but it’s a bit pricey and overbearing for a
child. That said, there’s still no substitute for the simplicity of Nintendo’s Switch Lite and its travel-friendly 5.5-inch display.
</item 51>

<item 52>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
“The Xbox wireless controller just works for me — it fits right in my hands, the texture feels just right, and it’s not too heavy” —Nicole Carpenter

Beyond PC compatibility, the Xbox controller boasts native compatibility on Steam Deck, plus recent versions of Android and iOS. So, whether you have a phone or
tablet, this controller can easily connect via Bluetooth. A growing number of mobile games support it, too, and it works like a charm with the Xbox Game Pass
game streaming app.

It can work on even more platforms if you purchase 8BitDo’s clever $19.99 USB Adapter 2. With it, you can use the Xbox controller on your Nintendo Switch. In
addition to Switch support, this USB adapter lets you easily connect your Xbox controller to other devices, like Mac, Raspberry Pi, and more.

The biggest downside to the Xbox controller might not bother you: It requires two AA batteries, though on a positive note, the battery life is around the
40-hour range, depending on usage.
</item 52>

<item 53>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 53>

<item 54>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$893 at Amazon

Sony InZone H3 $ 58 $ 100 42 % off $ 58 $ 58 $ 100 42 % off The InZone H3 is the wired (3.5mm / USB-A) model in Sony’s latest lineup of gaming headsets, which
are compatible with both PlayStation and PC. You can customize the audio profile of InZone headsets with the InZone companion app and Sony’s Spatial Sound
Personalizer. $58 at Amazon$60 at Best Buy

Elgato Stream Deck Mini $ 59 $ 80 26 % off $ 59 $ 59 $ 80 26 % off Elgato’s Stream Deck Mini is a smaller, six-key version of its popular macro pads. Its LCD
keys can be customized to offer quick one-button macros during a livestream, or it can simply control shortcuts on your computer and connect to your smart home.
</item 54>

<item 55>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Sony has been subtly expanding its roster — more subtly than Microsoft, at least — over the past three years, and it’s also made heavy investments in studios
like Epic Games and FromSoftware.

With this lineup, Sony is betting heavily on ongoing games, and it has 12 live-service titles in production right now, on top of Bungie’s Destiny franchise.
These include Haven’s Fairgame$ and a multiplayer Horizon title from Guerilla.

“By expanding to PC and mobile, and… also to live services, we have the opportunity to move from a situation of being present in a very narrow segment of the
overall gaming software market, to being present pretty much everywhere," Sony Interactive Entertainment president and CEO Jim Ryan said in 2022.

For the companies at the top, total domination is the goal.

Even still, Tencent is bigger than both Sony and Microsoft.
</item 55>

<item 56>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Sony DualSense controller

There’s another slice of our staff who swear by Sony’s DualSense, a modern controller that retains the symmetrical analog stick arrangement that PlayStation
controllers are known for. Compared directly to the Xbox controller, the DualSense has similar levels of compatibility (both are compatible with Android and
iOS, and feature built-in Bluetooth for support on PC). However, the DualSense does have key differences, as well as a few perks that help to justify its higher
$69.99 price.

In terms of using the DualSense on a PC, it’s possible to do, but it’s a little complicated. It can connect via Bluetooth for a wireless connection, but games
that don’t support the DirectInput controller API likely won’t recognize it as a controller. Plugging the controller into your PC might solve that issue, but
there are other ways to get around this barrier if playing wirelessly is important to you.

One is to rely on Steam, which supports the DualSense controller with its SteamInput API.
</item 56>

<item 57>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 57>

<item 58>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
For developers, this means the device has the same performance targets as the original and there’s no need to create games specifically for the new hardware. In
the end, the device weighs about 30 grams less than the original model, or the equivalent of five quarters.

Engadget

In use

Which leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the same monstrous size as the original: 11.7 inches long, 4.5 inches tall and
nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than the first version. This was one of my main complaints about the Steam Deck
when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks and feels absolutely ridiculous when I’m holding it. It’s just massive, no
matter how long my manicure gets. However, the OLED model feels more balanced than the original. It’s less top-heavy, and it does technically weigh less.
</item 58>

<item 59>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—CA

Related Viewfinder is puzzle game heaven

24. Humanity

Developer: tha

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Humanity looks like a modern art project, recalls memories of early PlayStation oddity, and comes (in part) from the team behind game of the future Rez
Infinite. But its most impressive feat is the way it gradually shifts genres, starting as a classic puzzle game, then taking on tower defense and shoot-’em-up
traits as it evolves into an all-out war. Think Braveheart, with a shiba inu general leading a troop of faceless, brainless low-polygon figures against a group
of angry, lightsaber-wielding, even lower-polygon foes. It’s the kind of design subtlety that stays in its lane yet builds on you, and before you realize it,
ends very differently than it began. Much like humanity itself? —ML

23.
</item 59>

<item 60>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 60>

<item 61>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
These days, Steam generally peaks at over 30 million, with valleys of 20 million. Fortnite’s valleys are closer to 3.8 million.]

Tim Sweeney says he plays Jellie in Fortnite. Image: Epic Games

Okay, let’s do a lightning round. What’s your favorite Fortnite skin?

I play as Jellie! Jellyfish. You know, the big tentacles? It’s just so cool!

Will Epic ever announce Unreal Tournament 3 X?

Forever in motion, the future is.

Why is Fortnite still not playable on Steam Deck?

If we only had a few more programmers. It’s the Linux problem. I love the Steam Deck hardware. Valve has done an amazing job there; I wish they would get to
tens of millions of users, at which point it would actually make sense to support it.
</item 61>

<item 62>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$400 at Amazon (refurbished)

Jsaux Mod Case for Steam Deck $ 24 $ 30 20 % off $ 24 $ 24 $ 30 20 % off A protective modular case system for Valve’s Steam Deck. It consists of a base case
with grippy texture, a built-in kickstand, a removable lid, and a rear mounting system that can attach a power bank using an adjustable strap or Jsaux’s add-on
cooling fan. $24 at Amazon

Jsaux’s 45W USB-C charger with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a
Nintendo Switch or just about any USB-C phone or tablet. It’s just $15.19 ($3 off) right now at Amazon.

with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a Nintendo Switch or just about
any USB-C phone or tablet.
</item 62>

<item 63>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
Fighting sticks can be really pricey. Thankfully, this fighting stick for the Xbox and PC is a great yet affordable option for getting into fighting games. $75
at Woot

The top-selling stuff on Polygon this week

The best entertainment deals this week

If you’re a fan of The Legend of Vox Machina series on Prime Video, you can pick up a two-disc Limited Edition vinyl LP of the soundtrack for just $37.56 on
Amazon, its lowest price to date. The soundtrack is packaged in a full-color gatefold sleeve and even contains exclusive tracks not featured in the series.

We recommend a lot of vinyl during our weekend posts, so it seems fitting that we should point you toward a solid record player. The Victrola Revolution Go
record player is one of the more unique suggestions from our 2023 Holiday Gift Guide and is currently on sale for around $125 in a variety of striking colors.
</item 63>

<item 64>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Google Pixel Tablet

Photo by Cherlynn Low / Engadget Google Pixel Tablet $409 $499 Save $90 The 11-inch Pixel Tablet comes with a charging speaker dock that allows it to double as
a smart home hub. This deal marks a new all-time low. $409 at Amazon

The 128GB Google Pixel Tablet is down to $409, which is $90 off and a new low. The 256GB model is also on sale for $509. While we think Samsung’s Galaxy Tab S9
works better purely as a tablet, Google’s 11-inch slate still gets you a sharp LCD display, a speedy Tensor G2 chip and an easy-to-hold design that weighs just
over a pound. Its big hook, however, is that it can attach to an included charging speaker dock and double as a Nest Hub Max-style smart display. Android still
isn’t as optimized for large screens as iPadOS, but if that extra functionality sounds useful to you, the Pixel Tablet is worth a look.
</item 64>

<item 65>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
One is to rely on Steam, which supports the DualSense controller with its SteamInput API. That makes Sony’s controller a first-class citizen in the eyes of
Steam, right next to the Xbox controller, and plenty of games should offer no-fuss DualSense support.

If you have PC games that you want to play on other services, like PC Game Pass, Epic Games Store, and more, you might experience some middling compatibility
with the DualSense. However, adding them as a “non-Steam game” within Steam can solve for this. To do this, click the “Add a Game” option in the bottom left
corner of the Steam client, then find the executable file (.exe) in your storage.
</item 65>

<item 66>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
One recent update saw it improve app launch times on both platforms. In the near future, Discord says people can expect a new “Midnight” dark mode the company
claims reduces battery consumption on devices with OLED screens, more functional notification tabs and a new search feature for the settings menu to make it
easier to find the exact option you’re looking to tweak.

Last but not least, the company says it will begin rolling out Premium App Subscriptions to eligible developers in the UK and Europe, following a launch that
began in the US a few weeks ago. On the subject of third-party apps, bots and plugins, the company said it’s in the process of exploring how to make those
accessible across nearly every part of Discord. It provided few details on the effort, but said the goal is to allow people to access their favorite apps and
bots without those experiences being restricted to select servers.

“We're experimenting with a few different things, but the goal is for developers to reach more people with the awesome experience they're building.
</item 66>

<item 67>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 67>

<item 68>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Also, the PS5’s interface for the Edge is intuitive, and it makes remapping controls and setting schemes the breeze that it should be.

While the Edge is compatible on PC, Sony and Steam haven’t developed drivers that let gamers take full advantage of its back paddles and swappable control
schemes.

Nintendo Switch Pro controller

The Switch Pro controller is another staff favorite. If it’s a great Nintendo Switch and PC controller you’re after, this is the pinnacle in terms of comfort,
durability, battery life, and other Switch-specific extras, but we’ll get to those a little later.

The Switch Pro controller can be used to play PC games. But like the DualSense, it’s not a plug-and-play affair unless you literally plug it in via USB. To use
it wirelessly, running Steam can help you out, as it officially supports the Switch Pro controller. Adding non-Steam games to your Steam library is a clever way
to ensure compatibility for games that support controllers.
</item 68>

<item 69>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It delivers on the promise of building a hacker-samurai in a neon-infused open world replete with futuristic heists, daunting choices, and striking characters.
Phantom Liberty and Patch 2.0 may not completely erase the memory of CD Projekt’s initial botched release, but they come pretty damn close. Three years after we
first set foot in V’s shoes, Cyberpunk 2077 has finally justified the hype. —M. Mahardy

14. Dead Space

Developer: Motive Studio

Where to play: PlayStation 5, Windows PC, and Xbox Series X

With The Last of Us on HBO and Resident Evil 4 back in the conversation, it’s already a banner year for survival horror. Motive Studio’s Dead Space remake is no
exception. Following in the footsteps of Capcom’s aforementioned title, the original Dead Space brought the third-person-action focus of Resident Evil 4’s
formula to a deteriorating ship in outer space.
</item 69>

<item 70>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Similar to Walmart and Amazon, be aware that items at Target shipped or sold by third-party retailers can result in longer lead times. Product listings usually
spell out your options for pickup, same-day delivery, and standard shipping — complete with estimated availability times — before you add them to your cart.
</item 70>

<item 71>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Much like humanity itself? —ML

23. Paranormasight: The Seven Mysteries of Honjo

Developer: Square Enix

Where to play: Android, iOS, Nintendo Switch, and Windows PC

Square Enix has plenty of mega-franchises to fill its time (and its coffers). This year, we have new entries for Octopath Traveler and Final Fantasy, along with
new Dragon Quest and Kingdom Hearts games in the not-so-distant future. Dayenu!

And yet, the publisher can’t help itself from bombarding us with surprising, interesting, sometimes great, often good-enough experiments. In 2022, we got an
English-language remake of lost gem Live A Live, the surprisingly enjoyable tactical RPG DioField Chronicle, a bonkers Final Fantasy spinoff featuring the
musical stylings of Limp Bizkit, and a pair of oddball card games lathered in lore from gaming’s best weirdo.
</item 71>

<item 72>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 72>

<item 73>
6 games to play first on your new Nintendo Switch (Polygon, 2023-12-24T15:00:00+00:00)
Each run where you die doesn’t feel like a loss — the story is simply that propulsive. There are so many weapon and buff combinations to try, you’ll be eager to
hop into your next playthrough, making the Switch’s portability crucial. Hades is for the Greek mythology nerds, but it’s also for everyone.

Stardew Valley

Solo developer and composer Eric Barone created both a game and a phenomenon with this one, which is inspired by the Harvest Moon series. In Stardew Valley you
inherit a farm from your grandfather, and move in as you’re eager to leave the corporate life behind. Farm, mine, fish, and make friends with the locals in
Pelican Town. It’s so easy to think to yourself “just one more day” and then lose hours, days, and months even to building out your farm, wooing a townsperson,
and completing community bundles. The multiplayer is also excellent, as you can share a farm with up to three other players.
</item 73>

<item 74>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
For instance, Amazon is offering some modest discounts on $50 Xbox Gift Cards, allowing you to pick up as many as you’d like for $45 each. Target has also
discounted three-month Game Pass Ultimate subscriptions to $44.99 (was $49.99), which allows you access a rotating library of excellent titles for the Xbox
Series X like Hi-Fi Rush, Cocoon, Venba, Starfield, and more.

The 8BitDo Arcade Stick for Xbox Series X is still available at Woot for $74.99, the best price yet for the $119.99 controller. This massive gamepad is a must
for fighting game fans, so whether you’ve recently picked up Street Fighter 6 or Mortal Kombat 1, or you’re looking forward to the launch of Tekken 8 in
February, now is the perfect time to pick up your new weapon of choice.

8BitDo arcade stick for Xbox $75

$120

38% off Prices taken at time of publishing. Fighting sticks can be really pricey.
</item 74>

<item 75>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
“I like the PS5 DualSense because it’s incredibly precise — using it, after coming from previous gen controllers, felt like putting on glasses and seeing the
world come into focus” —Nicole Clark

To get around using Steam entirely, you can download an app called DS4Windows along with some drivers, which ensures game compatibility on PC by emulating the
DualSense as an Xbox controller. The app gives you a host of settings to customize the DualSense’s button mapping, if you wish. And, in case you want to go down
the rabbit hole of figuring out which PC games support the DualSense’s haptics and adaptive trigger features (and the steps to getting them to work), the site
PCGamingWiki has you covered.

Related Here are the best PS5 accessories

The DualSense is the official controller for the PS5. In games that support its features, it can have realistic-feeling haptic feedback and gyroscopic motion
controls, and its triggers can provide tension to make certain actions feel more realistic. There’s no better controller for the PS5 than this one.
</item 75>

<item 76>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 76>

<item 77>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In the vein of Event Horizon, Sunshine, and Alien, Dead Space was a paragon for sci-fi horror in a confined and claustrophobic setting. Its remake has brought
that same vision to gorgeous new life, bringing quality-of-life changes and underappreciated updates (it has made several previously useless weapons into viable
tools in protagonist Isaac Clarke’s arsenal), making it hard to imagine ever going back to Visceral Games’ phenomenal original. —M. Mahardy

Related The Dead Space remake changes all the right things

13. Octopath Traveler 2

Developers: Square Enix, Acquire

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, and Windows PC

The first Octopath Traveler was one of those games that was as enjoyable to play as it was painful: enjoyable because so much of it kicked ass, but painful
because so much of it dragged the positive aspects down. In other words, it stood on the precipice of excellence, but couldn’t quite cross the line.
</item 77>

<item 78>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 78>

<item 79>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The improved 90 Hz refresh rate falls short of the Asus ROG Ally’s 120 Hz, though it’s worth considering that the higher the frame rate, the faster the battery
drain — assuming you can even spot the difference between 90 Hz and 120 Hz.

You know what’s more interesting than a list of technical specs? Actually seeing the screen. It’s gorgeous. On Sunday morning, I picked at Sega’s Like a Dragon
Gaiden: The Man Who Erased His Name while watching football and Christmas movies with my family. The new Yakuza game looks fantastic on high settings at 60 fps,
its night sequences flaunting the OLED’s dark blacks and the abundance of neon signage showcasing the HDR.

I got so distracted pulverizing bad dudes that it took me some time to notice the Steam Deck OLED’s most welcome feat. When playing AAA games on the original
Steam Deck, I had always kept the handheld plugged into an outlet. But with the OLED model, a couple of hours had passed before I realized I’d forgotten to
connect the charger.
</item 79>

<item 80>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
However, its biggest benefit is its Gulikit Hall effect analog sticks, which are resistant to developing drift over time by design. By comparison, the
potentiometer-based sticks used in the Ultimate 2.4g (and all of the other controllers mentioned so far) bear no such guarantee. There’s a possibility you may
never experience the issue firsthand, but the anti-drifting sticks alone might be a reason for some to consider buying the Ultimate Bluetooth.

The Ultimate Bluetooth is similar in a lot of ways to Nintendo’s Switch Pro controller, with its motion control support, rechargeable battery, and
Nintendo-style face button layout. However, it doesn’t have all of the same features. For instance, it doesn’t support NFC, so you can’t tap an amiibo on it,
and its haptics aren’t up to par with Nintendo’s. However, it’s one of a few third-party Switch controllers that can wake your console from sleep. If it’s
paired, press the home button, then shake the controller to awaken the Switch.

Update (Nov.
</item 80>

<item 81>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Viewfinder

Developer: Sad Owl Studios

Where to play: PlayStation 5 and Windows PC

At the heart of Viewfinder is a magic trick that never gets old: You take a photo of the environment and paste that perspective into the game world to create a
bridge or reveal a needed trinket. The experience of seeing reality distorted by your hands so easily is nearly on the level of thinking with Portals for the
first time. You’ll ponder a solution for minutes, sure it’s impossible and that the developers must have made a mistake, before you’re struck by a eureka moment
like a lightning bolt.

The game’s lesson on perspective extends to its narrative, which, much like the photos you use to manipulate the world, contains multitudes. Though it’s told in
fairly typical video game-y audio monologues that you uncover from the game’s trippy environments, the journey is powerful and topical. Just as differing
perspectives can allow for unique solutions, they can also obscure truths that lead us to deny the reality that’s right in front of us.
</item 81>

<item 82>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 82>

<item 83>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Like everything else in the Flower Kingdom, a few enemies shift and change in bizarre ways when the Wonder Flower is activated, like Hoppo, the bouncy, round
hippo equivalent that grows in size to create chaos in one of Super Mario Bros. Wonder’s many mesmerizing levels.

All of this weirdness, combined with Nintendo’s platforming expertise, makes Super Mario Bros. Wonder an entrancing romp alone, with friends, or with strangers
online. This is the freshest Mario has felt in decades, and it bodes well for the iconic plumber’s 2D future. —N. Carpenter

3. Alan Wake 2

Developer: Remedy Entertainment

Where to play: PlayStation 5, Windows PC, and Xbox Series X

There’s a body in the forest with a hole in its chest. The forest, like the body, is also missing its heart, and a woman must find both. In each, there are
pages. This is a story, broken into pieces, we learn.
</item 83>

<item 84>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 84>

<item 85>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—M. Mahardy

17. Dredge

Developer: Black Salt Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Dredge is a Lovecraftian horror experience masquerading as a simple fishing game.

The open ocean is filled with terrible creatures that can and will damage or destroy your boat. And each of the major islands you visit comes with its own evil
sea beasts that must be dealt with if you want to progress the story or fish peacefully. Dredge ultimately tells a dark parable about loss, and how obsession
can own you if you aren’t careful.

But what makes Dread so special — and one of the best games of 2023 — is that under its foreboding story and twisted environments is a fishing game that grows
more complex with every outing, centered around an upgrade system that feels amazing to progress through. As the dangers around you grow, so too does your
ship’s capabilities.
</item 85>

<item 86>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 86>

<item 87>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
This speaks to its comfort and value, and that Microsoft has released enough colors to satisfy varying tastes in design. It also helps that it’s usually around
$50 (depending on the color), and that it just works with most PC games. Regardless of the Windows PC game client, there are no extra steps for setting it up.

This controller is a refined version of Microsoft’s Xbox One wireless model, with added texture around its grips, plus a capture button in the middle. It can,
of course, connect to Xbox consoles, and to PCs that have Bluetooth. One universally positive change from the older version is that, if you want to connect via
a cable for wired play, its connector port is the more ubiquitous USB-C instead of Micro USB. It also has a 3.5mm headphone jack for private listening, though
it can only be used if your controller is wired into your PC, not wirelessly.
</item 87>

<item 88>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This investment alone means any time you buy a game built on Epic’s Unreal Engine, Tencent (and Sony) is getting a cut. Tencent is the biggest investor in
games, with thousands of tendrils across the industry — if you played something this year, Tencent was probably involved.

On a smaller scale, companies like Netflix and Devolver Digital have also dabbled in acquisitions recently. Devolver started buying studios in 2020, and it now
owns Croteam, Dodge Roll, Doinksoft, Firefly Studios, Nerial and System Era Softworks. Annapurna Interactive bought South African studio 24 Bit Games in
November. Netflix launched its Games division in 2021, and it’s already purchased four studios, including Oxenfree developer Night School and Alphabear company
Spry Fox.

Night School co-founder Sean Krankel told Engadget in June that the move to Netflix was a boon for the studio, providing financial security, a dedicated working
space and plenty of marketing support for its projects.
</item 88>

<item 89>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Hades is my most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like
Hades pops on the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same
settings (800p with a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing
Valve’s own estimates, though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing
Elden Ring with everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.

Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management.
</item 89>

<item 90>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 90>

<item 91>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$59 at Amazon

Storage for consoles and PCs

A massive 1TB SanDisk Extreme microSD card is now $89.99 $79.99 ($13 off) at Amazon. That can store a whole lot of games on a Steam Deck or Nintendo Switch.

Samsung Evo Select microSD card (512GB) $ 30 $ 40 25 % off $ 30 $ 30 $ 40 25 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $30 at Amazon

Western Digital WD_Black SN850X SSD $ 70 $ 180 61 % off $ 70 $ 70 $ 180 61 % off The WD_Black SN850X is one of the fastest M.2 drives around, with a sequential
read speed of up to 7,300 MB/s and a write speed of up to 6,600 MB/s.
</item 91>

<item 92>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
That's a $30 discount. Both the USB-C and Lightning models are on sale; the latter is compatible with Apple's iPhone 15 series as well as numerous Android
phones. We gave the Backbone a positive review last year, and we recommend it in our guide to the best gaming handhelds: It's more cramped than a traditional
gamepad, naturally, but it has all the requisite buttons to play modern games, it's extremely easy to set up and it works well for remote streaming from your
console or PC. If nothing else, it's a much better value than the PlayStation Portal.

PlayStation, Xbox and Switch video game sales

If you're all set on the hardware front, a truckload of PlayStation 5, Xbox Series X/S and Nintendo Switch games have dropped in price for Black Friday too.
Both Sony and Microsoft have rolled out holiday sales on their respective digital storefronts, while a number of physical games are down to new lows at Amazon,
Best Buy and others.
</item 92>

<item 93>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Even with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of a Wii U than a Vita. It’s a beast of a handheld,
and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready device. The new model even comes with an extra-long,
2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking station with the new Steam Deck, but Valve recently
released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only drives home the Wii U-ness of it all.

As far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll work on the handheld, with four levels:
unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy the on-screen keyboard, while verified
games have been fully optimized for portable play.
</item 93>

<item 94>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
It's our top pick in our best microSD card guide because it was the most consistent of the ones we tested, with some of the best overall speeds and all of the
important ratings that a good card should have: U3, V30 and A2. It'll be a great pick for your tablet, smartphone or Nintendo Switch, and it comes with a
full-sized adapter in case you need to use it with a camera or connect it to your computer.

Backbone One

The Backbone One is a well-built gamepad that snaps around the sides of your phone and works immediately with just about any game that supports controller
input. It’s also convenient for cloud streaming games from a console or PC. Right now both its USB-C and Lightning models are down to $70, which is $30 off.
That’s not an all-time low, but discounts on the device are relatively uncommon. We gave the One a positive review last year, and we currently highlight it in
our guide to the best handheld gaming systems.
</item 94>

<item 95>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It sports drift-free Hall effect sticks and is compatible with PC, Nintendo Switch, and Android.

8BitDo Ultimate Bluetooth Controller $ 56 $ 70 20 % off $ 56 $ 56 $ 70 20 % off The 8BitDo Ultimate Bluetooth Controller comes with its own charging dock,
features remappable controls, and is compatible with the Nintendo Switch, Steam Deck, and Windows PCs thanks to its Bluetooth and 2.4GHz connectivity. Unlike
the 2.4GHz-only version, it features drift-free Hall effect sticks. $56 at Amazon (black)

The RGB-ified PowerA Spectra Infinity Enhanced Wired Controller for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights,
it’s one of the cheapest controllers around to sport hair trigger lockouts.

for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights, it’s one of the cheapest controllers around to sport hair trigger
lockouts.
</item 95>

<item 96>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We strongly recommend OnePlus’ magnetic keyboard, which is also on sale for $100 ($50 off), however, if you purchase directly from the company's website, you
can get the keyboard bundled for free.

The 9th-gen iPad (8/10, WIRED Recommends) is our favorite iPad for most people, even though it's one generation behind. Aside from costing less than the current
10th-gen model, it is still compatible with the same accessories as the first-gen Apple Pencil. It retains the physical Home button with Touch ID at the bottom
of the screen.

Pixel Tablet Photograph: Google

Google's Pixel Tablet (7/10, WIRED Recommends) is part tablet, part smart display. It comes with a charging dock that doubles as a speaker, so you can pump out
the tunes and get answers from Google Assistant at a moment's notice. Take it off the charger and you get a full Android tablet experience.
</item 96>

<item 97>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.
</item 97>

<item 98>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Update (Nov. 17): Checked prices and links for accuracy.
</item 98>

<item 99>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 99>

<item 100>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
The Switch Pro controller is, unsurprisingly, a fantastic Switch controller. Like the Joy-Cons, it supports motion controls, HD rumble for more subtle, accurate
jolts of haptic feedback in games, and amiibo connectivity to redeem in-game perks in several games, including the upcoming The Legend of Zelda: Tears of the
Kingdom.

This might seem small, but it stands out as being one of a few wireless Switch controllers in existence that can wake the console remotely by holding down its
home button. Many require you to manually turn on the console, which can be a minor annoyance.

8BitDo Ultimate 2.4g controller

8BitDo has released a lot of controllers in recent years, but the name for its Ultimate controller is fitting. It has many of the features from its best
gamepads, plus some extras that should settle any sticker shock associated with its $49.99 price.

For starters, its asymmetrical analog stick layout, overall shape, and face button labels should feel right at home for people who love the shape of the Xbox
controller.
</item 100>
=== END ITEM id=7798ff2ba2607f70 ===

=== ITEM id=6977c2bd86d28841 ===
Gold sentences and facts:
Sentence 0: At the time, the music video for the song – which featured a woman with blonde hair - sparked rumours that he and Spears broke up because she allegedly cheated on him.
  Fact 0.0: At the time, the music video for the song featured a woman with blonde hair.
  Fact 0.1: The music video sparked rumours that he and Spears broke up.
  Fact 0.2: The rumours said the breakup happened because Spears allegedly cheated on him.
Sentence 1: Also, another thing that hints towards their breakup is Danis’s first press conference for the Logan Paul fight.
  Fact 1.0: Danis’s first press conference for the Logan Paul fight hints towards the breakup.
  Fact 1.1: Danis held a first press conference for the Logan Paul fight.

Retrieved items:

=== END ITEM id=6977c2bd86d28841 ===
