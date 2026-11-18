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

This file has 5 items.

=== ITEM id=b11b4a39cac1ec6e ===
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
</item 1>

<item 2>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.
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
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com
</item 5>

<item 6>
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
</item 6>

<item 7>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.
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
</item 9>

<item 10>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.
</item 10>
=== END ITEM id=b11b4a39cac1ec6e ===

=== ITEM id=69f7672a9e125904 ===
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
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 3>

<item 4>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.
</item 4>

<item 5>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 5>

<item 6>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 7>

<item 8>
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
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.
</item 9>

<item 10>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 10>
=== END ITEM id=69f7672a9e125904 ===

=== ITEM id=8de7bb8b27f6e790 ===
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
Here are a few of the most striking parts of the suit. While some of these allegations relate to practices from years ago, bear in mind that Meta (then
Facebook) has been publicly saying it doesn’t allow kids on the platform, and diligently worked to detect and expel them, for a decade.

Meta has internally tracked and documented under-13s, or U13s, in its audience breakdowns for years, as charts in the filing show. In 2018, for instance, it
noted that 20% of 12-year-olds on Instagram used it daily. And this was not in a presentation about how to remove them — it is relating to market penetration.
The other chart shows Meta’s “knowledge that 20-60% of 11- to 13-year-old users in particular birth cohorts had actively used Instagram on at least a monthly
basis.”

It’s hard to square this with the public position that users this age are not welcome. And it isn’t because leadership wasn’t aware.
</item 1>

<item 2>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
One stated goal, according to the lawsuit, was specifically to “grow [Monthly Active People], [Daily Active People] and time spent among U13 kids.”

It’s important to note here that while Meta does not permit accounts to be run by people under 13, there are plenty of ways it can lawfully and safely engage
with that demographic. Some kids just want to watch videos from SpongeBob Official, and that’s fine. However, Meta must verify parental consent and the ways it
can collect and use their data is limited.

But the redactions suggest these under-13 users are not of the lawfully and safely engaged type. Reports of underage accounts are reported to be automatically
ignored, and Meta “continues collecting the child’s personal information if there are no photos associated with the account.” Of 402,000 reports of accounts
owned by users under 13 in 2021, fewer than 164,000 were disabled.
</item 2>

<item 3>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 3>

<item 4>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.
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
</item 5>

<item 6>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.
</item 6>

<item 7>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 7>

<item 8>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
At the recommendation of its Oversight Board, Meta conducted a third-party due diligence report about the platform’s moderation during the May 2021 conflict.
The report found that Arabic content was flagged as potentially violating at significantly higher rates than Hebrew content was, and was more likely to be
erroneously removed. The report noted that Meta’s moderation system may not be as precise for Arabic content as it was for Hebrew content, because the latter is
a “more standardized language,” and suggested that reviewers may lack the linguistic and cultural competence to understand less common Arabic dialects like
Palestinian Arabic.

Has anything improved?

Meta committed to implementing policy changes based on the report’s recommendations, such as updating its keywords associated with dangerous organizations,
disclosing government requests to remove content and launching a hostile speech classifier for Hebrew content. Abukhater added that Meta has improved its
response to harassment, at least in comparison to other social media platforms like X (formerly Twitter).
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
</item 9>

<item 10>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 10>
=== END ITEM id=8de7bb8b27f6e790 ===

=== ITEM id=ad89503c395a5e1a ===
Gold sentences and facts:
Sentence 0: But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
  Fact 0.0: A woman says that people will look back and see this past year as a really important turning point for the role social media plays in our lives.
Sentence 1: The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.
  Fact 1.0: The Goellers are now seeking support again.
  Fact 1.1: The Goellers are launching the 'Save Lucas' campaign.
  Fact 1.2: The 'Save Lucas' campaign runs across social media.

Retrieved items:
<item 1>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.
</item 1>

<item 2>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“We were really seized by people saying they wanted something that felt safer, where they could speak openly without dogpiling,” says Oh. “And so we were
confident that was an important thing to hit. Are there other things that are equally important and a great user experience? Yes.”

“Twitter is you have real people…saying real things,” Oh continues. “At the expense of using a word that’s overused, ‘authentic.’ And our theory was always if
you can create a space where people really feel like they can let down their hair, then you would get that authentic content. We didn’t quite get there,” she
admits.

Cselle agrees, noting that, perhaps, Pebble should have opened up enough space for disagreement to happen, while still drawing a hard line on the most
disagreeable parts of running a Twitter-like platform.

“I think we could have maybe moved a little bit further towards that,” he says. “There’s a lot of nice things that are said on Pebble, but maybe not enough
breaking news.
</item 2>

<item 3>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.
</item 3>

<item 4>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.
</item 4>

<item 5>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Another app that spun out from laid-off former Twitter employees, Pebble (formerly T2), recently shut down after a year, growing to just 20,000 registered
users. Its CEO, Gabor Cselle, previously had exits to Google and Twitter, but accepted that Pebble simply wasn’t competitive enough to stay afloat. But while
Pebble sought to essentially clone Twitter, Spill’s team is building something that looks very little like it.

“People are looking for something new,” Terrell said. “I think things that have really defined, unique value propositions are going to win over the long term —
it might not be like there’s one winner-take-all.”

Earlier this year, TechCrunch covered the founding of Spill, piquing the interest of Kapor Capital, which ended up writing the company its first check.

“We had over 25,000 people join our waiting list in less than 24 hours from that article,” Terrell said.
</item 5>

<item 6>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.
</item 6>

<item 7>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
That’s one takeaway, at least, from the recent shutdown of Pebble (formerly T2), a Twitter clone that put trust and safety at the forefront of its development
process. The app never gained more than 20,000 users, and the founders now believe that its approach to moderation may have been correct, but it wasn’t a growth
driver. In other words, people may say they want to feel safer when participating online, but clearly, that isn’t their only need.

Mozilla’s instance, still in private testing, has a small team of moderators that will grow when the instance launches to the public in the coming months.

“It’s not just a commitment, it’s beyond foundational for us — that is the thing that is going to be true about this instance,” O’Hara says of Mozilla’s plans
to heavily moderate its space. Though Mozilla will also rely on technology to do some of the work, it wants to make sure the human moderators, “feel protected,
supported and respected for their work,” she adds.
</item 7>

<item 8>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
</item 8>

<item 9>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.
</item 9>

<item 10>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.
</item 10>
=== END ITEM id=ad89503c395a5e1a ===

=== ITEM id=d63cd2a66808f27d ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 0.1: The three categories are relevance, distance, and prominence.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The speaker admits that the speaker may have misinterpreted the evidence.
  Fact 1.1: The speaker maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 10>
=== END ITEM id=d63cd2a66808f27d ===
