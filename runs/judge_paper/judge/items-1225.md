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

This file has 1 items.

=== ITEM id=d84e27f5f062fcfd ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: If the debt continues to rise more than the economy, risks will increase.
  Fact 0.1: The government may 'crowd out' private investment.
  Fact 0.2: The government competes with the private sector for funds to service its debt.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said the Philippines worries about its debt-to-GDP ratio.
  Fact 1.1: President Ferdinand Marcos, Jr. said the Philippines' debt-to-GDP ratio is about 63%.
  Fact 1.2: President Ferdinand Marcos, Jr. said that 63% debt-to-GDP ratio is a little high for the Philippines and not ideal.
  Fact 1.3: President Ferdinand Marcos, Jr. said this to members of the US-ASEAN Business Council who visited Malacañang.
  Fact 1.4: The statement is reported by ABS-CBN on Aug. 9, 2023.

Retrieved items:
<item 1>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
During the Congressional budget hearing, Finance Secretary Benjamin Diokno optimistically provided updates on the government’s fiscal and revenue collection
performances and presented sources of funding for the national budget. Tax revenue is projected to increase from P3.5 trillion in 2023 to P6.5 trillion in 2028,
while non-tax revenues are expected to decline from P191 billion in 2023 to P184 billion in 2028, Diokno told the Philippine News Agency (pna.gov.ph, Aug. 10).

Diokno said the Finance department aims to gradually reduce net financing or new debt from P1.42 trillion in 2022 to P1.22 trillion in 2024, which will help
bring the country’s debt-to-GDP ratio to almost below 60% by 2025. And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%
and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr.
</item 1>

<item 2>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 2>

<item 3>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
</item 3>

<item 4>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
China's “decline,” as some have begun to call the conclusion of the country's four-decade-long economic boom, opens new opportunities for the Indian economy and
other developing and emerging countries.

Earlier this year, India reached another milestone when its population officially surpassed that of China, which had been the world's most populous country for
more than 300 years.

While China's shrinking, rapidly aging population is likely to impede economic growth and may curtail its geopolitical ambitions, India -- one of the world's
youngest countries, with a median age of 28.2 -- is poised to reap a huge demographic dividend, the article said.

But the driving force behind India's emergence as a major global power is its rapid economic growth. While India's gross domestic product is still smaller than
China's, the country is currently the world's fastest-growing major economy and is projected to account for 12.9 per cent of global growth over the next five
years, surpassing the United States' 11.3 per cent share, the article said.
</item 4>

<item 5>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
After the most aggressive monetary-tightening campaign in four decades, academics and economic practitioners are running autopsies on what could have prevented
the cost-of-living crisis and how to ensure the same mistakes won’t be repeated.

Markets have scrambled to price in high-for-longer interest rates, with a new war in the Middle East adding yet more risk to an already uncertain outlook
confronting central bankers as they gather for their penultimate meetings of a tumultuous year.

The policy navel-gazing is centering around three debates. How much flexibility central banks can allow in reaching their inflation targets, the effectiveness
of asset purchases in the policy mix, and the merits of monetary and fiscal coordination.

Bloomberg surveyed economists from around the world to gather views on those three debates. Their verdict: Central banks won’t break their economies in a rush
to hit inflation targets, QE will be used more sparingly in the future, and fiscal policy risks countering the work of monetary authorities.
</item 5>

<item 6>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Booms-turned-busts have been more abrupt and sensational, but rarely has an established industry so tied to the global economy lurched from historic profits to
below break-even levels more directly than the shipping lines that move 80% of the world’s merchandise trade have this year. After Covid’s massive demand shock,
the culprit now is too much supply.

“I’m certainly concerned about the next 24 to 36 months,” Rolf Habben Jansen, chief executive officer of Hamburg-based Hapag-Lloyd AG, said in an interview last
week. “We are going to see a downturn.”

Consider the tougher times facing A.P. Moller-Maersk A/S, the largest publicly traded container line. According to Bloomberg Intelligence credit analyst
Stephane Kovatchev, the Copenhagen-based company’s free cash flow, which reached $27 billion last year, may drop about 80% this year and could turn negative in
2024.
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
</item 7>

<item 8>
Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers (Fortune,
2023-12-06T21:36:17+00:00)
Executives and legislators discussed new regulations that would require the big banks to keep larger capital reserves. The big banks argued the new rules would
limit their ability to lend to lower and middle income buyers. In contrast, committee chair Senator Sherrod Brown (D-Ohio) disagreed and said any reduced
lending would be due to banks prioritizing riskier investments that have higher returns.

“You’d rather fund risky trading and derivatives bets than boring bread and butter small business lending,” Brown said in his opening remarks.

The indicted Menendez also framed some of the banks’ unpopular practices, like overdraft fees, as a decision rather than a necessity.

“Charging the fees is a choice—one that disproportionately harms Black and brown communities, Menendez said. Getting rid of overdraft fees would “change the
course of events for a significant part of [your] consumer base.”

Research points to the fact that overdraft fees disproportionately affect Black and Hispanic customers.
</item 8>

<item 9>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
In Fiscal Year 2022, this amounted to P74 billion.

Moreover, under the Universal Health Care Law, PhilHealth receives 50% of the government share from the income of the Philippine Amusement Gaming Corp. (PAGCOR)
as well as 40% of the Charity Fund of the Philippine Charity Sweepstakes Office (PCSO). In 2022, this amounted to P188 billion.

In like manner, in addition to the school vouchers funded under the Expanded Government Assistance to Students and Teachers in Private Education Act, we suggest
that PhilEd be allocated a share of the taxes imposed on our extractive industries. This is based on the argument that the depletion of our natural resources
can only be justified if the taxes collected will enhance our human resources.

Like PhilHealth, PhilEd could also be mandated to receive a percentage from the collection of documentary stamps which have recently been raised.

Finally, PhilEd can be given access to the Special Education Funds of the local school boards of the Local Government.
</item 9>

<item 10>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
The PE (price to earnings ratio) of a stock is 18. Is that cheap or expensive? Well, it depends. On what? Many things! Welcome to the world of relative
valuation.

The findings from research done a decade earlier by Aswath Damodaran indicated that around 85 per cent of valuation done by equity research analysts and 50 per
cent of corporate M&A transactions were based on relative valuation. So, what exactly is this relative valuation?

It is the art of attempting to determine the fair value of a stock based on how it is valued relative to many factors. So, whether the PE of a stock is 15 or 18
or 30, the only way you can make an assessment of whether it is cheap or expensive is by analysing this one number — against its growth prospects, peer company
multiples, industry trading range, historical average PE, the prevailing interest rates, etc.
</item 10>

<item 11>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
But both sides are still waiting for more action. "The current trend in China-U.S. relations is one of easing," said Shen Yamei, director of the department for
American Studies and an associate research fellow at the China Institute of International Studies. "This easing is a relaxation of the atmosphere," she said in
Mandarin, translated by CNBC. "No actual changes have occurred." She pointed out, however, that the establishment of many new communication channels means there
is much to look forward to.

Export controls

During this week's meeting, Shen expects the Chinese side to bring up U.S. export controls and investment restrictions. The Biden administration has restricted
U.S. companies from selling high-end tech, primarily in semiconductors, to Chinese companies and sought to curb U.S. investments in such Chinese tech. U.S.
Commerce Secretary Gina Raimondo in a trip in August had "said no" to China's requests to reduce the controls and called the them "matters of national
security."
</item 11>

<item 12>
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
</item 12>

<item 13>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 13>

<item 14>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.
</item 14>

<item 15>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
), the Oil Price Stabilization Fund, the Bangko Sentral ng Pilipinas, and the Central Bank-Board of Liquidators. It is allocated for the implementation of
various government programs and projects, the operation of government offices, payment of salaries of government employees, and payment of public debts. These
expenditures are classified by expense class, sector and implementing unit of government (dbm.gov.ph).

The sources of appropriations of the annual budget are: 1.) new general appropriations legislated by Congress for every budget year under the General
Appropriations Act (GAA); and 2.) existing appropriations previously authorized by Congress. Under the Constitution, Article VI, Section 29, no money can be
withdrawn from the Treasury except in pursuance of an appropriation made by law.

The DBM calls a budget balanced when revenues match expenditures or disbursements. When expenditures exceed revenues, the government incurs a deficit which may
result in the following situations: a.)
</item 15>

<item 16>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
It’s important to consider the past track record and management team before making investment decisions. Consultation with investment advisors who are
knowledgeable in this area can undoubtedly be beneficial.

The author is President & Head, Nuvama Wealth
</item 16>

<item 17>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
— Ryan Selkis 🪳 (@twobitidiot) October 2, 2023

Shameful that Michael Lewis is defending SBF. “They had a great real business. If no one had cast aspersions on the business, if there hadn’t been a run on
customers deposits, they’d still be making tons of money” – Michael Lewis That's not true. They are not a bank, they're… pic.twitter.com/vRKBbgzITe — Dan Held
(@danheld) October 2, 2023

But even so, Noff is defensive: She also believes crypto is no worse than the stock market, which can also be manipulated and mishandled.

“Even when you identify corruption, I don’t think that means we don’t believe in the system in question,” she added.

Yet Lewis’s picture is a little less stark from the looks of the 60 Minutes episode.
</item 17>

<item 18>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
We recently presented a paper at the 5th Paderanga-Varela Memorial Lecture entitled “Alternate Service Delivery Systems: Health and Education.” In that lecture,
we looked at the Department of Health (DoH) and the Department of Education (DepEd) as service delivery systems, delivering healthcare and education
respectively.

We concluded that the service delivery system of the DoH is far superior to the DepEd for two main reasons. The first is that the DoH has devolved its hospitals
to the local government units while the DepEd has not. Secondly, the DoH has spun off its health financing into the Philippine Health Insurance Corp.
(PhilHealth) while the DepEd has not. In this article we argue for the creation of the Philippine Education Development Fund (PhilEd).

When Isidro Consunji, President of DMCI Holdings, Inc. was asked by his officers what the role of his Chief Financial Officer (CFO) would be, he gave a
three-word answer, “Fund my dreams.”
</item 18>

<item 19>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
But a spread of the crisis to the whole Middle East can risk trade through this route, which accounts for 20 per cent of global oil trade passage. If Iran is
embroiled into this conflict, then trade with landlocked Central Asia, which is a big agricultural producer, raw materials and minerals like copper, cotton,
uranium, gold, natural gas, gets disrupted. Central Asia also hosts a network of road, rail, and air routes that link China, Europe, and South Asia.

Impact on Indian markets

In the above scenario, supply disruption will lead to higher inflation and central bankers will have to adjust their policies and hold rates higher for longer
than anticipated. We may witness a scenario where rates are held higher till the year 2025. Higher crude oil bill and higher global rates will prompt the RBI to
also remain hawkish in its approach with regards to rates and liquidity measures. EBITDA margins of India Inc. may again get impacted on account of higher input
costs and higher freight and fuel costs.
</item 19>

<item 20>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
The world can ill-afford another full-fledged war, when economies are already battling high inflation and interest rates, an ongoing Russia-Ukraine war and now,
a slowing China. This is precisely why many countries are trying to restrict any escalation in the ongoing Israel-Hamas conflict. The non-escaltion of the
current hostilities seems to be the base case presumed by the Indian stock market presently, going by the market movement in the last week.

The US treasury secretary also stated that the current conflict is not likely to significantly impact global economic outlook. History too shows that the these
conflicts tend to have limited impact on markets. We studied the history of conflicts between Israel and Hamas, and found that these tend to have limited impact
on global financial markets, across crude oil, US 10-year yields, US equities and market volatility (see charts).

However, as far as trade goes, there may be some impact. In 2022-23, Israel’s share in exports from India stood at 1.87 per cent.
</item 20>

<item 21>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
The geopolitics of US-China tensions, chip wars, decoupling and “friendshoring” seem, in particular, to be working significantly in Japan’s favour. Asia-focused
global investment is rotating out of China and into Tokyo stocks, real estate and other assets, just as large parts of corporate Japan are engaging more
actively with shareholders than at any time in the past.

China, after years of stunning growth and investor attention, is very rapidly losing that sheen. Its economy is growing more slowly and the expected
post-pandemic rebound is proving elusive. On top of that, Beijing’s policymaking — with its sudden crackdowns on entire industries, like video games and private
tutoring, has generated a series of nasty surprises — and the unshakeable fear that others could occur at any time.

“You are seeing more interest in Japan from investors, who were primarily invested in China, as China has slowed and some of its economic policies have been
confusing and opaque,” says Kirk Neureiter, president of Fidelity Management & Research Japan.
</item 21>

<item 22>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
The ECB faces an extra legal burden on bond holdings that comes with operating in a currency union of 20 countries. Concerns around illegally financing
governments and debt mutualization have already landed the central bank in court several times.

Mixing Policies

Low interest rates and large-scale QE programs allowed treasuries to borrow on the cheap to finance stimulus campaigns, protecting labor markets, businesses and
consumers from collapse. But the spending blowout throughout and since the pandemic — part critical emergency funding, part political need to show an
all-hands-on-deck approach in crisis — contributed to the latest outbreak in inflation.

While the same kind of pulling in the same direction is needed to restrain demand, many governments are concerned that if they tighten policy too hard, voters
will kick them out and replace them with populists or extremists. That’s reviving questions about whether central banks can deliver price stability all on their
own.
</item 22>

<item 23>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
India's strong FDI inflows have been boosted by large inflows of investments from global technology MNCs such as Google and Facebook that are attracted to
India's large, fast-growing domestic consumer market, as well as a strong upturn in foreign direct investment inflows from manufacturing firms.

Overall, India is expected to continue to be one of the world's fastest growing economies over the next decade. This will make India one of the most important
long-term growth markets for multinationals in a wide range of industries, including manufacturing industries such as autos, electronics and chemicals to
services industries such as banking, insurance, asset management, health care and information technology, it added.

Brahma Chellaney, professor of strategic studies at the New Delhi-based Center for Policy Research, wrote in a recent article that as India's geopolitical,
economic and cultural clout grows, so does its global footprint.
</item 23>

<item 24>
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
</item 24>

<item 25>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.
</item 25>

<item 26>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
How can this larger deficit be financed?

"It’s reasonable to argue that besides the hope of financing part of the deficit through aid from the US (President Biden approved $8 billion military aid this
week, H. S.), and that will happen, it’s permissible for a country at war to raise its fiscal deficit by what is seen as the right amount. This is in order to
ensure that the economy functions properly, to carry out rehabilitation actions, and to prepare for recovery and a return to growth in the near future."

How high should the deficit go?

"I mean a deficit of 3% of GDP, or 5%, or 8% - there are no sacred numbers. As was said during the Covid pandemic, the state has to do whatever it takes to
achieve these goals. Nevertheless, it’s important to make sure that the growth in expenditure and in the deficit is temporary, until economic activity gets back
to normal.
</item 26>

<item 27>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
What Bloomberg Economics Says… “A long period of galloping price gains, and fears that the last yards back to target could be most painful for workers, have
reignited the debate about whether central banks should aim for a higher rate of inflation. That’s a conversation worth having. But for monetary policymakers,
the imperative of retaining credibility means the right time for it is after inflation is back at target, not before.” — Tom Orlik, chief economist

Rethinking Targets

So long as people believe prices will get back toward 2%, central bankers have some leeway in deciding how aggressive they need to be in pursuing that goal.

Economists covering 16 of the world’s most important central banks say policymakers will allow more time to bring inflation back to target if it means less
damage to their economies. The Bloomberg special survey also shows that a sizable minority sees them going even further, accepting price pressures that are
either slightly too strong or too weak — as long as expectations remain anchored.
</item 27>

<item 28>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 28>

<item 29>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Billionaire investor Paul Tudor Jones told CNBC the current geopolitical environment was the “most threatening and challenging” he’d ever seen following Hamas’s
attack on Israel over the weekend and predicted the US will enter into a recession early next year. The shekel regained its footing as the central bank duelled
short sellers to contain the market fallout from Israel’s conflict with militant group Hamas. European gas prices jumped, with Finland on high alert as it
suspects a gas pipeline leak in the Baltic Sea was caused by a deliberate act of destruction, fuelling concerns about the safety of Europe’s energy
infrastructure. Base metals sank amid fresh fears over property turmoil in China. Tweet of the day

Quote of the day “This is as strong a level of confidence in the market that we have seen since before COVID,” said The Leasing Agency group managing director
Terence Yeh as the return of international students and a growing number of apartments in inner-city Melbourne provides a shot in the arm for CBD hospitality
businesses.
</item 29>

<item 30>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
That’s reviving questions about whether central banks can deliver price stability all on their own.

“If we were designing optimal policy arrangements from scratch, monetary and fiscal policy would both have a role in managing the economic cycle and inflation,
and that there would be close coordination,” Philip Lowe said in his last speech as Australian central bank governor in September.

Economists surveyed by Bloomberg predict fiscal policy will somewhat counteract the Fed’s efforts to rein in inflation in the US.

“It’s true that there are circumstances where working hand in hand and supporting each other has proved helpful,” ECB President Christine Lagarde told a panel
discussion in June at the institution’s annual economic forum.

Fed Chair Jerome Powell, who sat to her right, signaled he wasn’t ready to rely on that kind of cooperation. “Our assignment is to deliver price stability kind
of regardless of the stance of fiscal policy.”

Central bankers warn that any failure to scale back fiscal spending risks coming at the cost of yet higher interest rates.
</item 30>

<item 31>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For the energy sector in particular, amendments to the PSA are envisioned to help raise the capital needed to speed up the energy transition of the country to
renewables.

In addition, the Regional Comprehensive Economic Partnership (RCEP), a trading bloc that encompasses the ten members of the Association of Southeast Asian
Nations (ASEAN) and the ASEAN Plus One Free Trade Agreement (FTA) partners Japan, China, South Korea, Australia, and New Zealand, is already in force for the
Philippines, helping ease market access through trade and investment rules and supporting global and regional supply chains. The Philippines can potentially
position itself as a regional manufacturing hub if the right incentives and policy measures are put in place to encourage local and foreign investors to
participate. Together, trade and investment are expected to play an important role in attaining economic transformation, the second goal of the PDP.

HUMAN CAPITAL AND INFRASTRUCTURE

The ability of people to take advantage of market opportunities arising from investment rests on human capital.
</item 31>

<item 32>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.
</item 32>

<item 33>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
Beijing considers Taiwan part of its territory, with no right to independently conduct diplomatic relations. The U.S. recognizes Beijing as the sole government
of China but maintains unofficial relations with Taiwan, a democratically self-governed island. While speaker of the U.S. House of Representatives in August
2021, Nancy Pelosi became the highest-ranking U.S. official to visit Taiwan in 25 years. The trip prompted Beijing to suspend talks on climate with the U.S.,
one of the few areas of potential cooperation.

Areas of cooperation

The Biden administration has said the U.S. is in competition with China, while looking to ensure that it "does not veer into conflict." "The Biden-Xi meeting
might include a pledge to cooperate or establish a new formal bilateral working group on safe use of artificial intelligence," Teneo's Wildau said.
</item 33>

<item 34>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Just like PDIC which is funded from the premiums paid by the banks, PhilEd will fund this program from the premiums by the educational plan companies. PhilEd
will be ideally suited to do this as it could swap the failed educational plans with school vouchers from its repository of school vouchers.

The third dream is based on the realization of Filipino school graduates that their learning must continue even as they leave their alma maters. For in this
modern age, the fount of personal knowledge must continuously be replenished. In healthcare, we have the Health Maintenance Organizations (HMOs) which
continuously monitor and minister to our health. We need Lifelong Learning Organizations (LLOs) to continuously monitor and minister to our learning. HMOs exist
in the Philippines. LLOs do not. In partnership with the international development agencies and the private sector, PhilEd will create this industry. The
international development agencies will provide the technical assistance grants that will create the developmental and regulatory framework for the industry
while the private sector will organize the LLOs.
</item 34>

<item 35>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Emerging market economies with large volumes of USD denominated debt are especially vulnerable as they could be caught in a vicious circle of their external
debt (in local currency terms) ballooning due to strong DXY and their servicing ability being impaired by tepid economic activity owing to high rates. Even
advanced economies have felt the heat, and the ECB etc. have maintained a hawkish narrative, Jain said.

Another developing story seems to be the greenshoots of recovery in China backed by domestic consumption in the Golden week, and tourism related to the Asian
Games. For now, these seem to buttressed on stimulus and base effects as China is now fully open.

The long-term trajectory remains to be seen. In any case, for now this has provided a floor for commodity prices, especially base metals. Fickle commodity
prices will also be victims of fresh geopolitical tensions in West Asia, which has pushed up Brent crude. Food prices could also play truant with climate change
affecting cropping area worldwide: sugar and rice prices are booming, the note said.
</item 35>

<item 36>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 36>

<item 37>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
It’s again near its highest level since 2007.

Loading Treasury bonds are seen as some of the safest investments possible, and when they pay higher yields, investors are less likely to pay high prices for
stocks and other riskier investments. That’s a big reason why the S&P 500 dropped 4.9 per cent in September to drag what had been a big gain for the year down
to 11.7 per cent. Treasury yields have been climbing sharply as Wall Street accepts a new normal where the Federal Reserve is likely to keep interest rates high
for longer. The Fed is trying to push still-high inflation down to its target, and its main tool of high interest rates does that by trying to slow the economy
and hurting prices for investments. The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut
interest rates next year by less than it earlier expected.
</item 37>

<item 38>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
That may weigh on the company’s bonds, he wrote in a research note on Friday.

Read More: Maersk Slumps After Forecasting Weak Global Trade Until 2026

Over the past 10 days, Maersk, Hapag-Lloyd and closely held CMA CGM SA of France — all top-five players that together control about one-third of the world’s
container capacity — said they’re cutting costs as some fear the slump will last at least through 2024.

Some executives are urging against price wars, which contributed to a wave of consolidation and at least one major bankruptcy in the years leading up to the
pandemic.

“Each actor will have to be responsible to ensure that the market remains reasonable amid rates that are relatively low,” CMA CGM Chief Financial Officer Ramon
Fernandez told reporters Friday. “Price wars after a while hurt not only those who start them but everyone.”
</item 38>

<item 39>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com
</item 39>

<item 40>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
The IMF. in its latest update. has said that India is one of the large emerging economies that has been doing better than expected.

The IMF said India has been doing better for a while now and that India is, in fact, one of the growth engines in the world economy at this point.

The IMF said important divergences are appearing. The slowdown is more pronounced in advanced economies than in emerging markets and developing economies. Among
the advanced economies, the US has been revised up, with resilient consumption and investment, while the euro area has been revised down, as tighter monetary
policy and the energy crisis took a toll.

There is divergence also among emerging markets and developing economies, with China facing growing headwinds, while Brazil, India. and Russia are revised up.

After two years of rapid economic growth in 2021 and 2022, the near-term economic outlook for India is for continued rapid expansion during 2023-24, underpinned
by strong growth in private consumption and investment, S&P Global Market Intelligence said in a recent update.
</item 40>

<item 41>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Kovatchev said what’s emerging is a standoff between the strong and the weak. “The bigger companies such as Maersk and Hapag-Lloyd have the cash to wait and
focus on cost-cutting, as opposed to aggressive capacity reductions — for now,” he said. “It all boils down to supply, demand and who will blink first.”

Of course, the flipside of shipping’s pain are lower costs for the manufacturers and retailers that own the cargo being transported, which ultimately helps
central bankers tasked with bringing down still-elevated inflation across many developed economies.

“A couple years ago, it was double-digit inflation in goods prices and maybe a 4% or 5% increase in services,” said Phil Levy, chief economist at Flexport Inc.,
a San Francisco-based digital freight-forwarding company. “To the extent you were getting inflationary pressure from goods, or in a very tight goods market —
that has disappeared.”
</item 41>

<item 42>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
She recommends they be ready to react preemptively, even when inflation hasn’t yet spun out of control.

They may be called into action soon on that front, should an escalation in the conflict in the Middle East hit oil deliveries.

When the next big global slowdown comes, though, flexibility may be needed the other way. Europe’s eight-year experiment with negative rates ended with mixed
reviews last summer as to whether it was all worth it.

The Bank for International Settlements argues that there’s room for greater tolerance for moderate shortfalls even if they’re persistent, because “low-inflation
regimes, in contrast to high-inflation ones, have self-stabilizing properties.”

Rethinking Quantitative Easing

With a more flexible approach to those 2% targets, monetary policy after the 2008 financial crisis would have looked very different in many parts of the world.
</item 42>

<item 43>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
The acceleration of foreign direct investment inflows into India over the past decade reflects the favourable long-term growth outlook for the Indian economy,
helped by a youthful demographic profile and rapidly rising urban household incomes.

India's nominal GDP measured in USD terms is forecast to rise from $3.5 trillion in 2022 to $7.3 trillion by 2030. This rapid pace of economic expansion would
result in the size of the Indian GDP exceeding the Japanese GDP by 2030, making India the second largest economy in the Asia-Pacific region.

By 2022, the size of the Indian GDP had already become larger than the GDP of the UK and also France. By 2030, India's GDP is also forecast to surpass Germany,
the research said.

The long-term outlook for the Indian economy is supported by a number of key growth drivers. An important positive factor for India is its large and
fast-growing middle class, which is helping to drive consumer spending.
</item 43>

<item 44>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
For security reasons, Israel has already told Chevron to shut down its offshore energy platform.

The Middle East crisis brings back memories of the Arab oil embargo of 1973, when OPEC cut oil to supporters of Israel. It used to account for 50 per cent of
the oil production back then. Although non-OPEC production has picked up and OPEC contribution is lower at 32 per cent, it is still sizeable to take oil prices
higher. A 10 per cent rise in crude price will result in global inflation being 0.4 percentage points higher next year.

The war may also impact Suez Canal trade, which is a key maritime route between Asia and Europe and accounts for 12 per cent of global trade by value and much
larger by volume; also 5 per cent of world oil trade.

As things stand today, trade may not yet be impacted through the Strait of Hormuz, which lies between Oman and Iran.
</item 44>

<item 45>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
The Department of Budget and Management (DBM) defines the government budget as “the financial plan of the government…which shows what its resources are, and how
they will be generated and used over the fiscal period. It refers to the income, expenditures and sources of borrowings of the National Government (NG) that are
used to achieve national objectives, strategies and programs.”

Section 22, Article VII of the Constitution states that: “The President shall submit to the Congress within 30 days from the opening of every regular session,
as the basis of the general appropriation bill, a budget of expenditures and sources of financing including receipts from existing and proposed revenue
measures.”

The National Government (NG) budget is the totality of the budgets of all government entities, namely the National Government, the non-financial
government-owned and -controlled corporations (which usually includes only the 14 major GOCCs), government financial institutions (GFIs), local government units
(LGUs), the social security institutions (the Government Service Insurance System or GSIS, the Social Security System or SSS, the military pensions, PhilHealth,
etc.
</item 45>

<item 46>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
U.S. President Joe Biden shakes hands with Chinese President Xi Jinping as they meet on the sidelines of the G20 leaders' summit in Bali, Indonesia, November
14, 2022. Kevin Lamarque | Reuters

BEIJING — After another rocky year of U.S.-China tensions, the two countries' presidents are set to meet this week in person for the second time since Joe Biden
took office. It will be a rare summit before the U.S. presidential election cycle kicks off in earnest. Taking a tough stance on China, the second-largest
economy in the world, has become one of the few areas of bipartisan agreement. Biden plans to run for reelection. "The focus will be on expanding dialogue in
order to low[er] tail risks in the relationship and prevent a crisis that neither leader is looking for," said Michael Hirson, head of China Research at 22V
Research. "Flashpoints such as Taiwan and the South China Sea need to be managed carefully," he said.
</item 46>

<item 47>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“I would urge you to evaluate what we are doing in my country, look at the underlying strength of our economy and our plans for the future and then invest in
Japan,” Kishida said in a recent speech at the Economic Club of New York.

On the face of it, the Tokyo conference guests have been told by various experts that the stars do appear aligned in Japan’s favour, both on its own merits and
in comparison with what is happening around it. But, because so many global funds have diverted expertise away from Japan in recent years, they are concerned
that they need to know more, as a matter of urgency.

“There is heightened interest in Japan after what has been, for many, a long period of not being focused on that country,” says Paul Hitchens, the head of the
Japan research department at Jefferies, which hosted a major investor conference in Hong Kong.
</item 47>

<item 48>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Iron & steel have also witnessed accelerated long-term loan disbursements in the past six months. Infrastructure is witnessing an increase in credit flow with
67 per cent of the respondents indicating an increase in long-term loans as against 57 per cent in the previous round.

The survey suggests that the outlook on expectation on growth of non-food industry credit over next six months is optimistic with 42 per cent of the
participating banks expecting non-food industry credit growth to be above 12 per cent (as compared to 36 per cent in the previous round).

Rajan Jain, Head, Credit Research, SBI Caps, said in a note that whether the yield curve dynamics mean the US is moving to a structurally higher-rate economy
vs. post GFC normal remains a moot question, with important implications for investment demand and system risk.

These movements have had a cascading effect on global economies, which have seen routs in their respective bond markets, besides a fresh round of currency
depreciation.
</item 48>

<item 49>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
“Price wars after a while hurt not only those who start them but everyone.”

Read More: Shipping Giant CMA CGM Warns Against Price War as Profit Plunges

Such concern stems from a combination of economic forces: Goods demand is returning to pre-pandemic levels just as supply is rising in the form of new, bigger
ships. It can take two to three years to build a container ship, which typically operate for about 25 years. So timing their launch and retirement with the ebbs
and flows of the business cycle is inherently difficult.

To manage capacity in the short term, the main tools at the carriers’ disposal are canceling individual voyages or suspending services entirely on trade lanes
where demand is weak. In prolonged slumps they can also let charter contracts expire, idle some ships or sell old ones in the scrap steel market.

Kovatchev said what’s emerging is a standoff between the strong and the weak.
</item 49>

<item 50>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Any other metric could be wrong and misleading.

For example, price by revenue or price by EBITDA is incorrect. Why? How do you differentiate between two companies A and B — both having same market cap of ₹100
and similar in all means except for debt levels — same revenue (₹100) and same gross margin and same EBITDA margins of 20 per cent and thus same EBITDA of ₹20.
But A has zero debt, while B has debt of ₹10. Since revenue or sales and EBITDA are same, if you do a market cap/revenue or Price /Sales, or marketcap/EBITDA,
both companies will have the same valuation.

However, let’s assume depreciation is ₹10, interest cost is 10 per cent and taxes are zero. This means the net profit of A is ₹10, while that of B is ₹9.
</item 50>

<item 51>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Enterprise value multiples

Enterprise value refers to total value of a company/business. This total value as perceived by the market is represented by the sum of its market cap + net debt
(includes minority interest). What works in the case of EV is that it helps in doing a capital-agnostic analysis of a company. Whether a company is funded only
with equity, or a combination of equity and debt, the total value of the business is the same. Sources of funding don’t matter.

Using EV, the value of equity (share price) is derived from the value of the business. Since the capital structure of companies varies significantly, an EV
approach enables valuing them on similar terms. One can arrive at the value of the business first and then subtract net debt, to arrive at the equity value. The
advantage with EV-based metrics is that you have more options to value a company.

Bear in mind, valuation accuracy is better the lower down the P&L and cash flow statement one can get to.
</item 51>

<item 52>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
QE effectively swaps long-term borrowing costs for short-term ones. What’s been a lucrative deal for taxpayers when official interest rates were low has now
turned into a disastrous trade.

The clearest depiction of the problem is in the UK, where the BOE secured taxpayer indemnity for any losses on QE. Over the next decade, it estimates, its
purchases will cost the government over £200 billion ($243 billion).

And policymakers have little experience in unwinding their balance sheets, where small mistakes can trigger big market turbulence.

The Fed experienced some of that when it tried to shrink bond holdings between 2017 and 2019. More recent efforts to reduce portfolios have progressed rather
smoothly, partially because central banks have amassed so much debt over the years that they’re far away from any thresholds that would trigger a squeeze.

But the fact that they’re treating quantitative tightening as a technical adjustment rather than a part of their efforts to conquer inflation raises questions
about the future use of a tool that’s only trusted to work one way.
</item 52>

<item 53>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Large global funds are not the only ones to have had their interest in Japan piqued, says Keith Truelove, head of global markets at UBS Securities Japan.

“There is still a lot of dry powder on the sidelines,” he observes. “There is a lot of interest in Japan from family offices and from high net worth individual
investors, who value wealth preservation more highly than institutional investors — it’s really taking off now.”

“They took their time to decide if it was safe to invest in Japan again, because they need to get confidence in the idea that the Japan story will be more
sustainable this time,” Truelove adds. He notes that Japan has offered investors a number of false-dawns over the years, and the bar for deciding that ‘this
time is different’ remains high.

One of the most significant changes, say asset managers, is the fundamental shift in the way global funds now look at China.
</item 53>

<item 54>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
They also want elected officials to put in place policies that help deliver sustainable growth.

“A change in mindset needs to happen,” said Agustin Carstens, the former governor of the Bank of Mexico who’s now the general manager of the BIS. “Growth needs
to depend less on fiscal and monetary policy, it should depend more on structural policies.”
</item 54>

<item 55>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Park and her team noted that there are no “noticeable signs” of excess housing development today like there were back then, and households aren’t nearly as
burdened by mortgage debt. Household mortgage debt represented roughly 65% of U.S. consumers’ disposable income in the second quarter, compared to 100% before
the GFC. And the ratio of Americans’ mortgage debt to their real estate assets—also called loan-to-value—was just 27% in the second quarter, compared to over
40% in 2008 and roughly 50% in 2010, Bank of America’s data shows.

New legislation was also enacted since the GFC to help prevent worst-case scenarios in the housing market. One of the most obvious effects of these new laws is
that there are far fewer risky adjustable-rate mortgages today.
</item 55>

<item 56>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
You may have missed Corporate leaders caught underpaying staff should expect zero sympathy from new Fair Work Ombudsman Anna Booth, who believes chief
executives and board directors are directly responsible for any failures to pay correct wages. With Bloomberg/Reuters
</item 56>

<item 57>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Congress enacted Republic Act 7875, known as “The National Health Insurance Act of 1995,” which created PhilHealth. We are hoping some members of Congress would
file a Universal Education Financing Law which would create PhilEd.

Dr. Victor S. Limlingan is a retired professor of AIM and a fellow of the Foundation for Economic Freedom. He is presently chairman of Cristina Research
Foundation, a public policy adviser and Regina Capital Development Corp., a member of the Philippine Stock Exchange.
</item 57>

<item 58>
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
</item 58>

<item 59>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
The banking system is stable, and there is enough liquidity in the system. The behavior of the investment community in Israel, including the financial
institutions, indicates high confidence in the economy’s resilience and financial strength. It’s important to maintain a level-headed, correct economic policy
that will preserve this ‘asset.’"

What about the economic cost of the war?

"As with any war, the current war has and will have direct and indirect impacts. The war will substantially raise defense spending in the coming years, for
procurement of equipment, ammunition, and advanced systems in accordance with the new needs. Another direct expense is payment for the reserve duty days of
those mobilized. Given current interest rate levels around the world, the cost of servicing public debt as a proportion of GDP will rise. Besides all this,
there will be a decline in state revenues from taxation and a rise in the fiscal deficit."

How can this larger deficit be financed?
</item 59>

<item 60>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
Psagot doesn’t specialize in long-term savings, in which most of the money is invested outside Israel, but mainly in investment in the local market, in products
such as portfolio management, mutual funds, and a little brokerage. These are investment channels that are very vulnerable when Israel takes a hit. Finally, the
war that broke out on October 7 was a blow to the finance industry, certainly in the short term."

Now, Group Psagot is setting out to raise NIS 220 million debt, to be secured by the company’s shares in subsidiary Psagot Investment House. To that end, the
company’s management has held meetings in the past few days with several financial institutions, to present to them the investment house’s activities and
business plans. The sum that Group Psagot seeks to raise is intended both to cover the balance of its debt to Mizrahi Tefahot Bank, and to replace NIS 120
million in marketable bonds due for redemption in the coming eighteen months.
</item 60>

<item 61>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Throughout September, in Tokyo, New York, Hong Kong and London, the world’s biggest investment banks hosted a series of conferences for the world’s biggest
investors, all gravitating around the same theme: is Japan really back?

And, according to both the attendees and hosts of those events — which were run by JPMorgan, Goldman Sachs, Bank of America, Mizuho, BNP Paribas, and Jefferies
— the answer is now clearer than it has been in a very long time. Japan is back, but global investors may take a little more persuasion that it is truly back
for the duration.

So, this week, prime minister Fumio Kishida will make his strongest ever push for a bet on Asia’s biggest advanced economy — as BlackRock founder Larry Fink,
and other heads of global funds, gather in Tokyo for a series of events to attract investment.
</item 61>

<item 62>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Furthermore, PMS is available in two options: discretionary and non-discretionary. In the former case, the portfolio managers make all decisions; in the latter
case, the investor makes all decisions.

3. Market-linked Debentures (MLD)

For HNIs looking to invest a part of their investible surplus in debt, MLDs can be a prudent option. MLDs are non-convertible debt instruments whose returns are
tied to a specific financial market benchmark, such as the Nifty 50, the BSE Sensex, or the 10-year G-Sec. The maturity periods of MLDs can vary from one year
to five years.

Though the minimum investment for MLDs is ₹1 lakh, making it easier even for small investors to invest in them, they are more popular with the HNIs owing to the
complex structure of these debentures. MLDs, unlike traditional fixed-income securities, do not provide periodic interest payments before maturity. Income
multipliers and principal protection are common features of equity-linked MLDs.
</item 62>

<item 63>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
While Indian interest rates were always higher, investors in equities benefited from the global investors’ hunger for yields. That advantage is diminishing
today. This is not a recommendation on the stocks, but pointers on factors to consider to make your relative valuation process more foolproof.

With discipline and experience, the process will keep getting better.
</item 63>

<item 64>
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
</item 64>

<item 65>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Equity multiples

If you check the balance sheet of any company, the liabilities side is primarily made up of two line items — shareholder funds and borrowings. To the contrary,
a Profit and Loss statement has many line items — starting with revenue, followed by variable costs and then fixed costs and taxes. The only claim that equity
shareholders have for distribution or for appropriation amongst themselves is what is left after paying the taxes — net profit.

So, if you are doing relative valuation based on equity multiples, it is important to ensure you divide the equity – ie price per share or market cap of the
company by line items that apply to equity shareholders.

Based on this, there are only 3 items that can be attributed to equity shareholders — net profit from P&L statement, book value or shareholder funds from the
Balance Sheet, and the free cash flows to equity (FCFE) from the cash flow statement. Any relative valuation multiples will be linked to these three items one
way or the other. Any other metric could be wrong and misleading.
</item 65>

<item 66>
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
</item 66>

<item 67>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 67>

<item 68>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
PhilEd will be the “bastonero” or project manager, managing the project as well as mobilizing government support and assistance.

FAPE or the Fund for Assistance to Private Education was a part of the Special Fund for Education set aside by the American government in 1963 from the surplus
funds authorized by the War Damage Act of 1962 and organized under Executive Order 156 issued by President Ferdinand Marcos on Nov. 5, 1968.

From the initial seed money of $6,154,000 or P215 million (based on the 1968 peso to dollar exchange rate), the FAPE fund in 55 years had barely grown to P277
million. In contrast the trustees managing the fund have expanded their services, basically as service provider to the DepEd. As a consequence, it has been spun
off as a separate organization, the Private Education Assistance Committee (PEAC).
</item 68>

<item 69>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.
</item 69>

<item 70>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
The seventeenth round of the FICCI-IBA survey was carried out for the period January to June 2023. A total of 24 banks including public sector, private sector
and foreign banks participated in the survey. These banks together represent about 79 per cent of the banking industry, as classified by asset size.

The Indian economy posted an impressive 7.2 per cent growth in the financial year ended March 2023 and 7.8 per cent in first quarter of current fiscal year.
Credit growth also continued to rise, signalling robust demand conditions in the Indian economy, as well as banks' improved appetite towards retail borrowers.

The health of the banking sector has witnessed an encouraging turnaround, marked by healthier bank balance sheets and gross NPA ratio at a decade low.

The survey findings show that long term credit demand has seen continued growth in sectors such as Infrastructure, textiles and chemicals. food processing and
metals.

Iron & steel have also witnessed accelerated long-term loan disbursements in the past six months.
</item 70>

<item 71>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
I’m also sure that there are various budget items that were approved that don’t support economic growth, and that don’t need to be implemented in the new
reality. Such a policy will be accepted with complete understanding by investors overseas and in Israel, and also by the credit rating agencies."

What else do you propose on the economic policy plane?

"Despite the uncertainty over how the war will develop, alongside an emergency budget, this is the time to start preparing a general economic plan with a
strategy for rehabilitating the economy after the war. It’s important to start planning professionally and in detail all aspects of rehabilitating the
settlements that have been damaged or destroyed in the south, rehabilitation of infrastructure, the health system, and the civilian systems that need
attention."

How do you propose doing this?

"One possibility is to task the Bank of Israel and the Ministry of Finance Budgets Division with coordinating the work, with help from the various research
institutes and economists, engineers, and professionals in academic institutions and in the private sector.
</item 71>

<item 72>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
The surplus budget policy is important to encourage economic growth,” the DBM said.

The exhortations of the national budget managers must be quoted word for word to educate the non-economists and questioning common people about how a surplus
budget is necessary versus a deficit budget:

“The less the government borrow from the public, the lesser the pressure on interest and inflation rates and the more funds are made available in the financial
market. Such funds may be used by businessmen to build factories, hire workers, buy equipment and open more employment opportunities. By keeping more funds in
the hands of the private sector rather than competing for credit, the government helps make financing available for families who want to own homes, buy cars, or
support their children’s education. The government also needs to generate a budget surplus to repay the huge debt it has accumulated over the years. The
reduction of the national budget debt will correspondingly lessen government’s requirements for interest and principal payments. This becomes important
particularly during periods of rising interest rates and unstable exchange rates” (dbm.gov.ph).
</item 72>

<item 73>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Trillions of dollars, euros, yen and pounds of asset purchases did little to raise prices in the face of global disinflationary forces until governments used
the money they raised to stuff cash into consumers’ pockets during Covid lockdowns.

But that’s also been blamed for distorting financial markets. Episodes such as the Silicon Valley Bank blow-up are seen by some as a direct result of central
bank reserves creation under QE, along with regulatory and supervision failures.

Only 40% of economists surveyed predict central banks will use QE the same way as they did before. A quarter expect them to deploy it more sparingly, about 30%
see its only role going forward as a tool to address financial-stability concerns and a small minority doesn’t see it being used again at all.

There are other problems with bond-buying that may affect how it’s used in the future. QE effectively swaps long-term borrowing costs for short-term ones.
</item 73>

<item 74>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
A massive strike by Israel in Gaza, potentially killing civilians in collateral damage, may attract more of the Middle East countries, including the de-facto
leader Saudi Arabia to raise serious issues against Israel on humanitarian grounds. It will surely slow down the deal between Saudi and Israel that the US was
facilitating. Besides Gaza, clashes have already been ongoing along Israel’s borders with Syria and Lebanon. If Iran gets dragged into the conflict, on account
of claims of funding Hamas, this could embroil the entire Middle East in crisis.

In this scenario, we see a major upward pressure on world inflation, on account of disruption in crude oil and world trade logistics.

Possible disruption in crude and trade logistics

Even before the Hamas attack, Saudi Arabia and Russia further cut oil production, in an already tight supply market. Iran is a big incremental supplier of oil,
contributing 3 per cent to the global supply and any escalation risks taking energy prices higher. For security reasons, Israel has already told Chevron to shut
down its offshore energy platform.
</item 74>

<item 75>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Japanese prime minister Fumio Kishida will appear this week at a series of events aimed at attracting foreign investment © AFP via Getty Images

There are distinctive catalysts for that, he adds — noting a recent push by the Tokyo Stock Exchange for companies to improve their capital efficiency, and the
high profile decision of Warren Buffett to invest heavily in five of Japan’s largest trading houses.

Hitchens says: “Is there some healthy scepticism? Yes. However, global investment firms know that they need to invest and build their offerings in Japan to meet
the growing appetite, and that they need to pay attention to an equity market at a 33-year high.”

Topix, Japan’s main stock index, has risen more than 25 per cent this year, outperforming almost all the world’s developed market indices and, as some
conference attendees admit, creating a “fear of missing out” on a rally that may have significantly further to go.
</item 75>

<item 76>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“Deployment of capital to China has been a bit more risky in terms of eventual outcome of investment decisions, given that lack of clarity.” But interest in
Japan has not only come about because of the push away from China, he adds.

Japanese companies are increasingly aligning themselves with the interests of shareholders in a way that once seemed unlikely. Part of that, he says, is
generational. Companies that replace their chief executive on five or seven year cycles now have leaders that reached their positions after the establishment of
Japan’s stewardship code (in 2014) and corporate governance code (2015). Many have a very different view of the role of a chief executive than their
predecessors did.

“I think that Japan, in terms of the unifying message that companies and the Tokyo Stock Exchange are sending out, now gives a lot more confidence about where
they are deploying capital,” says Neureiter. “So people looking at the region are finding Japan incrementally more attractive.
</item 76>

<item 77>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Noel P. Rabaja the strategy and transactions (SaT) service leader of SGV & Co.
</item 77>

<item 78>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Likewise the role of PhilEd would be to fund the education dreams of all Filipinos especially the neediest and most deserving.

Filipinos have three dreams in education. The first is to enroll in a chosen course in the school of their choice. This dream will be funded by the School
Vouchers Program which will now be transferred from DepEd to PhilEd.

The second dream, especially for Filipino parents, is to save enough money to fund the college education of their children, hence the prevalence of pre-need
Educational Plans. Unfortunately, most of the pre-need companies went into bankruptcies causing financial ruin and emotional distress to their plan holders.
There has been a massive market failure and so government must intervene. But intervention should not involve the government offering educational plans but
rather in insuring these plans. Just as the Philippine Deposit Insurance Corp. (PDIC) insures the bank deposits of Filipinos, so will the PhilEd insure their
educational plans.
</item 78>

<item 79>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Loading Wall Street has dealt with such shutdowns in the past, and stocks have historically been turbulent in the run-up to them, according to Lori Calvasina,
strategist at RBC Capital Markets. After looking at the seven shutdowns that lasted 10 days or more since the 1970s, she found the S&P 500 dropped an average of
roughly 10 per cent in the three months heading into them. But stocks managed to hold up rather well during the shutdowns, falling an average of just 0.3 per
cent, before rebounding meaningfully afterward. Wall Street is also contending with higher oil prices, shaky economies around the world, a strike by US
autoworkers that could put more upward pressure on inflation and a resumption of US student-loan repayments that could dent spending by households.

On Wall Street, the vast majority of stocks fell under such pressures, including 90 per cent of those within the S&P 500.
</item 79>

<item 80>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Gross direct tax revenue experienced a substantial increase of 18 per cent y/y up to October 9, allaying concerns of fiscal slippage, the note said.
</item 80>

<item 81>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
We have a great deal of information about loss of growth and product in various episodes in the past, such as the Yom Kippur War, the Lebanon War, two
intifadas, various military operations, and the Covid pandemic.

"We see this is a completely different episode, however. This is a new situation that has no precedent as far as the damage to the economy is concerned. There’s
a whole spectrum of possible scenarios, from a short war on the southern front only, to the nightmare scenario of a prolonged war with additional sectors
involved. We are only in the fifth day of the war, and it’s to be hoped that the picture will start to become clearer."

Which industries will be hit by the war?

"A substantial decline can be expected in private consumption, particularly of services. This is not just because of the damage to output and income, but also
for psychological reasons. Distress, uncertainty, the possibility of prolonged reserve duty by some of the public - all these things will affect consumer
sentiment.
</item 81>

<item 82>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
This is what economist Larry Summers termed as Ketchup economists/economics. It was his sarcastic take on some of the highly paid economists and finance
professionals who are concerned only with inter-relationships between prices of different financial assets. Based on this approach, they conclude whether the
price of an asset independently is efficient or not. If two bottles of ketchup sell for twice as much as one bottle (except for minor differences traceable to
transaction costs), the market is efficient.

The risk in this – it ignores the aspect of whether one bottle of ketchup is priced rationally. Nobel Laureate Paul Krugman once explained how the US housing
bubble that precipitated the global financial crisis reminded him of ketchup economists. Before purchasing a house during the bubble, buyers carefully compared
prices with ‘prices of other houses’, but ignored whether the overall level of a home price made sense.

How to avoid pitfalls

So, what can you do to avoid pitfalls while using relative valuation? Here are a few pointers.
</item 82>

<item 83>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Over the years, PEAC has evolved into co-implementing externally funded programs, namely the DepEd’s Education Service Contracting (ESC), Teachers’ Salary
Subsidy (TSS), the SHS Voucher Program (SHS VP), the In-Service Training (INSET), and the Bayanihan for Basic Education (BBE) as well as implementing internally
funded programs of assistance for private education in the areas of training, school quality assurance, school improvement programs, and grants programs.

In sum, the PEAC has been an effective instrument to operationalize “complementarity” between private and public schools as provided in the Philippine
Constitution. It has also been an incorruptible partner of the education agencies in efficiently co-implementing national subsidy programs for students and
teachers in private schools.

We propose that FAPE/PEAC structure and organization be moved to PhilEd. Operationally this would mean folding into PhilEd, the roles and responsibilities of
FAPE and PEAC as well as their financial resources.
</item 83>

<item 84>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Hence, they tend to rise and fall lower than their peers. Investors with stock and mutual fund portfolios can diversify with hedge funds to reduce volatility.

e) Performing credit funds: Many operating companies cannot raise funds from banks and NBFCs due to their size or credit ratings. This is where performing
credit funds come into play. These funds specialise in credit risk assessment and, after thorough due diligence, offer credit to such companies at relatively
higher rates, resulting in higher overall returns than AAA/AA-rated bonds. These funds are attractive for investors seeking higher returns on their fixed-income
portfolio, which includes debt funds and AA/AAA-rated bonds.

f) Infrastructure Funds: These funds invest in long-term infrastructure assets owned by PSUs or private companies, such as roads, power transmission lines, and
solar power plants. Most infrastructure projects provide essential societal services and thus have intrinsic value. These assets generate cashflows in the form
of annuities backed by Central and State governments, lowering the risk of future cash flows significantly.
</item 84>

<item 85>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
It’s important to talk to the private sector, and to understand in every place and every industry what the main damage is and what they need in order to get
back to normal. It should be done professionally, preferably with no politicization of the recommendations."

What do you think the Bank of Israel should do in its next interest rate decision, on October 23?

"It’s still to early to tell how the war will affect the rate of inflation in the short and medium term. Just as there are new inflationary factors, such as the
depreciation of the shekel, there will also be factors working in the opposite direction, given the expected decline in private consumption, tourism, and other
items.

"As with fiscal policy, here too a country at war is allowed to deviate temporarily from the set inflation target. At the moment there is no surge in inflation
expectations, and the investment community is aware of the existence of index-linked assets and the exchange rate that can assist in hedging inflation risk.

"As for cutting the interest rate, it’s too early to consider that.
</item 85>

<item 86>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
embassy in Beijing. China's Ministry of Commerce last week announced it was gathering information in an effort to address unequal treatment of foreign
businesses in China versus domestic ones — a longstanding business complaint. However, on the cultural front, the three remaining giant pandas in the U.S. on
loan from Beijing returned to China last week due to an expiring contract. China has lent pandas to countries around the world as a diplomatic tool.
</item 86>

<item 87>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
may again get impacted on account of higher input costs and higher freight and fuel costs.

Rural India is a key engine that is not firing at the moment. Valuations also are rich and do not offer margin of safety in the near to medium term. Near-term
attractiveness of equities will reduce, as investors will demand a higher risk premium. The Indian stock market will likely correct materially, as FIIs look to
withdraw funds from risk assets and seek safe havens in US Dollar, US treasuries offering high yields and Gold.

Sectors which will be negatively impacted include Petrochemicals, Oil Marketing, Paints, Packaging, Textiles and Chemicals.

The writer is Group President & Head - Institutional Equities, YES Securities India

SHARE Copy link

Email

Facebook

Twitter

Telegram

LinkedIn

WhatsApp

Reddit
</item 87>

<item 88>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.
</item 88>

<item 89>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
"As for cutting the interest rate, it’s too early to consider that. The current interest rate is at the right level for the state of the economy, and together
with the policy on exchange rates, it contributes to ensuring financial stability," Leiderman concludes.

Published by Globes, Israel business news - en.globes.co.il - on October 12, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 89>

<item 90>
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
</item 90>

<item 91>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
Chinese Vice Premier He Lifeng also raised the issues during preparatory meetings with U.S. Treasury Secretary Janet Yellen in San Francisco on Nov. 10,
according to state media. "Aside from Taiwan, export controls are Beijing's top concern, but there is no political space in Washington roll back existing
controls," Gabriel Wildau, managing director at consulting firm Teneo, said in a note. "The immediate aftermath of the [Biden-Xi] meeting is likely to mark a
cyclical high point for bilateral relations," he said. "The key question is whether this high point extends into a plateau or whether political pressures
trigger a new cycle of deterioration," he said. "As previously discussed, the period since June has offered a window of opportunity to stabilize relations;
following the meeting, this window may close." Taiwan is set to hold its presidential election in January, and a more pro-independence winner could stir more of
Beijing's ire.
</item 91>

<item 92>
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
</item 92>

<item 93>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“So people looking at the region are finding Japan incrementally more attractive. I’ve never seen such a consensus around what needs to be done for Japanese
companies to improve capital efficiency and returns on capital,” he adds.

Efforts by Japanese businesses to demonstrate better governance are aided by government initiatives to make Tokyo more attractive for foreign asset managers.
Kishida has promised major reform of the country’s $5tn asset management industry, which would ease regulations that had made it difficult for foreign and new
players to enter.

The government plans to expand the Nippon Individual Savings Account (Nisa) tax-free investment scheme from next year in a bid to unlock $14tn in household
financial assets that has long sat in cash and bank deposits.

“This is completely different from before,” says Stefanie Drews, president of Nikko Asset Management. “In the past, allocation to Japan had been tactical but
this time it is strategic and that is why it is much more meaningful.”
</item 93>

<item 94>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
“After extensive deliberations and negotiations, the House of Representatives on Wednesday (Sept. 27) approved on third and final reading the P5.768-trillion
national budget for next year, just before Congress adjourns for a month-long break,” the Press and Public Affairs Bureau of Congress announced on its website,
congress.gov.ph. The 310-strong legislative chamber voted 296 against three, with zero abstentions, to approve House Bill (HB) No. 8980 or the 2024 national
budget on third and final reading. Almost unanimously.

House Speaker Ferdinand Martin G. Romualdez noted that the budgeting process was marked by rigorous discussions, particularly concerning confidential and
intelligence funds, which the House meticulously scrutinized to ensure accountability and responsible utilization. He lauded “the timely passage of the 2024
General Appropriations Bill (GAB),” stating that it not only “demonstrated the commitment of the House to upholding its constitutional role but also underscored
its dedication to serving the Filipino people with transparency and accountability.”
</item 94>

<item 95>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 95>

<item 96>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Folding FAPE/PEAC into PhilEd would mean the transfer of the funds of FAPE amounting to P277 million as of May 31, 2022 as well as the assets and liabilities of
PEAC amounting to P567 million as of May 31, 2022.

The DepEd programs presently being managed by PEAC and lodged in the DepEd budget such as the Educational Service Contracting (ESC) of P9.3 billion in School
Year 2021-2022, the Senior High School Voucher Program of P20.3 billion in School Year 2021-2022, and the Teachers Salaries’ Subsidy of P793 million in School
Year 2021-2022 will now be lodged in PhilEd.
</item 96>

<item 97>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
And it’s totally fine to be an investor relying on relative valuation to make your savings work for you, provided you approach it the right way and ensure
adequate risk management. In relative valuation, the broader you assess the valuation, ie assess valuation multiple against more parameters, the better the
perspective you get.

In simple words, one way to understand the difference between absolute valuation (explained in last week’s edition) and relative valuation is this – absolute
valuation is made up of an interesting story that is backed by credible and validated numbers, while relative valuation is made up of an interesting number
backed by a credible and validated story.

What are those numbers in relative valuation? Every number in relative valuation is the result of a number A divided by a number B. For example, PE is price of
the stock divided by its earnings. These are all numbers that are used to compare against similar metrics of other stocks and assess the merits of the stock.

Before we proceed on using relative valuation to pick stocks, it is important to understand that you can do relative valuation using equity multiples or
firm/enterprise multiples.
</item 97>

<item 98>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When other assets get more attractive one must assess scope for valuation change in equities as well. For example, interest rates and bond yields in many
developed markets, including the US, are at their highest levels since 2007. This means investors may have to assess current valuations not just against last 10
years’ valuations, but of prior periods as well.

Why? Near zero interest rates and low bond yields made stocks with earnings yield of mere 1-2 per cent attractive for an FPI — for example stocks such as Asian
Paints, Nestle India and ABB India which today trade at PE of 65, 83 and 85 times respectively. This implies earnings yield of 1.5 per cent for Asian Paints and
1.2 per cent for Nestle and ABB. Such earnings yields of these stocks were fine for FPIs to hold when compared to zero interest income in their domestic market.
But now, with US bond yields offering 5 per cent, these may not be so attractive even when considering the growth prospects.
</item 98>

<item 99>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
In this challenging environment, Indian real GDP growth, estimated to be 6.2 per cent y/y in real terms in FY24, is resilient. High frequency indicators suggest
strong construction and infrastructure activity, and soaring urban demand.

Manufacturing production has been unusually strong in Q2FY24, especially in infrastructure-oriented structures such as cement and steel. Power supply grew
vastly in Q2 after a tepid Q1, helped by a dry August 2023 and buzzing industrial activity.

Both the Union and States have front ended capex in H1FY24, and this remains the engine of expansion. Capital expenditure was up 48 per cent y/y for the Union
in the same period. Reports suggest capex of major states was also up 45 per cent y/y in 5MFY24.

Carrying forward this capex in H2FY24 would be important to sustain the multiplier. The fiscal deficit for the 5MFY24 stood at 36% of the FY24BE.
</item 99>

<item 100>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
"This translates into a stalemate—and, in fact, spiraling tensions—that go even further than the typical 'security dilemma,' in which each side takes steps to
defend itself which in turn generate insecurity for the other, who then responds in kind," the report said.

The immediate aftermath of the [Biden-Xi] meeting is likely to mark a cyclical high point for bilateral relations Gabriel Wildau Teneo

Sentiment began to improve over the summer after U.S. Secretary of State Antony Blinken finally made a high-stakes visit to Beijing in June, followed by visits
from several other senior officials. In early October, U.S. Senate Majority Leader Chuck Schumer and five other U.S. senators representing both the Republican
and Democratic parties had an 80-minute meeting with Chinese President Xi Jinping. But both sides are still waiting for more action. "The current trend in
China-U.S.
</item 100>
=== END ITEM id=d84e27f5f062fcfd ===
