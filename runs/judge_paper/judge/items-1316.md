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

=== ITEM id=2915c95baca805ba ===
Gold sentences and facts:
Sentence 0: The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut interest rates next year by less than it earlier expected.
  Fact 0.0: The Fed's main interest rate is at its highest level since 2001.
  Fact 0.1: The central bank indicated last week that it may cut interest rates next year.
  Fact 0.2: The central bank indicated last week that any cut to interest rates next year may be smaller than it earlier expected.
Sentence 1: Stocks surged through the week on rising hopes that the Federal Reserve is finally done with its market-crunching hikes to interest rates, in order to get inflation under control.
  Fact 1.0: Stocks surged through the week.
  Fact 1.1: Stocks surged on rising hopes that the Federal Reserve is finally done with its market-crunching hikes to interest rates.
  Fact 1.2: The Federal Reserve's hikes to interest rates were market-crunching.
  Fact 1.3: The Federal Reserve's interest rate hikes were intended to get inflation under control.

Retrieved items:
<item 1>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.
</item 1>

<item 2>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Lower yields have been one reason prices for cryptocurrencies have been rising recently. Excitement about a possible exchange-traded fund tied to bitcoin, which
would open it to new kinds of investors, has also helped send it above $US43,000 ($65,600) recently. The surge of interest helped Robinhood Markets report a
roughly 75 per cent jump in trading volumes for crypto during November from a month earlier. It also said customers added about $US1.4 billion in net deposits
during the month, and its own stock rose 10.3 per cent.

On the losing end of Wall Street was Take-Two Interactive, which slipped 0.5 per cent after a trailer for its highly anticipated Grand Theft Auto VI video game
said it’s coming in 2025. That was later than some analysts expected.
</item 2>

<item 3>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Emerging market economies with large volumes of USD denominated debt are especially vulnerable as they could be caught in a vicious circle of their external
debt (in local currency terms) ballooning due to strong DXY and their servicing ability being impaired by tepid economic activity owing to high rates. Even
advanced economies have felt the heat, and the ECB etc. have maintained a hawkish narrative, Jain said.

Another developing story seems to be the greenshoots of recovery in China backed by domestic consumption in the Golden week, and tourism related to the Asian
Games. For now, these seem to buttressed on stimulus and base effects as China is now fully open.

The long-term trajectory remains to be seen. In any case, for now this has provided a floor for commodity prices, especially base metals. Fickle commodity
prices will also be victims of fresh geopolitical tensions in West Asia, which has pushed up Brent crude. Food prices could also play truant with climate change
affecting cropping area worldwide: sugar and rice prices are booming, the note said.
</item 3>

<item 4>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Trillions of dollars, euros, yen and pounds of asset purchases did little to raise prices in the face of global disinflationary forces until governments used
the money they raised to stuff cash into consumers’ pockets during Covid lockdowns.

But that’s also been blamed for distorting financial markets. Episodes such as the Silicon Valley Bank blow-up are seen by some as a direct result of central
bank reserves creation under QE, along with regulatory and supervision failures.

Only 40% of economists surveyed predict central banks will use QE the same way as they did before. A quarter expect them to deploy it more sparingly, about 30%
see its only role going forward as a tool to address financial-stability concerns and a small minority doesn’t see it being used again at all.

There are other problems with bond-buying that may affect how it’s used in the future. QE effectively swaps long-term borrowing costs for short-term ones.
</item 4>

<item 5>
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
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Treasury Wine Estates (down 1.9 per cent) and Woolworths (down 1.2 per cent) dragged the consumer staples sector (down 1.1 per cent) lower. Pilbara Minerals
(down 2.1 per cent) and Qantas (down 1.5 per cent) were also among the biggest large-cap decliners. The lowdown Stronger iron ore prices helped to bolster the
local bourse on Monday but not enough to offset the prospect of higher-for-longer interest rates ahead of the Reserve Bank’s meeting on Tuesday. Iron ore
heavyweights advanced after the iron ore price increased 1.4 per cent overnight with lithium miner Liontown resisting the downward momentum of lithium prices,
which dropped to the lowest level in two years.

Most sectors traded lower but interest-rate sensitive sectors including healthcare and consumer companies were among the weakest amid expectations that the RBA
will keep rates on hold in October. The National Australia Bank has forecast another rate hike to 4.35 per cent in November.
</item 7>

<item 8>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Iron ore giants Rio Tinto (down 1.5 per cent) and BHP (down 0.7 per cent) slipped after the iron ore price fell 1.1 per cent. Aristocrat Leisure (down 3.7 per
cent) dragged the consumer discretionary sector (down 0.3 per cent) lower.

The lowdown IG Australia market analyst Tony Sycamore said the ASX received some relief from a rally on Wall Street following the release of
cooler-than-expected US economic data which helped cap US bond yields. “Much has been written about the unrelenting rise in long-term US interest rates
threatening prospects of a soft landing and raising doubts around equity market valuations,” he said. “This was stopped in its tracks overnight following the
release of a considerably weaker employment, and ISM services purchasing managers index data topped off by a 5.6 per cent fall in the price of crude oil to
below $85 per barrel.”
</item 8>

<item 9>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
This is what the market has priced in today.”

The local gains came after most stocks slipped on Wall Street overnight, although the US market stayed near its highest level in 20 months following a mixed set
of reports that kept alive questions about whether the world’s largest economy can pull off a perfect landing where it kills high inflation, but avoids a
recession. The S&P 500 edged down by 0.1 per cent for its first back-to-back loss since October. The Dow Jones slipped 0.2 per cent and the Nasdaq composite
rose 0.3 per cent. Stocks were down more sharply in Asia amid worries about the health of China’s economy, the world’s second largest. US stocks and Treasury
yields wavered after reports showed that US employers advertised far fewer jobs at the end of October than expected, while growth for services businesses
accelerated more last month than expected. Hope has been rising on Wall Street recently that the US economy is slowing from its recent hot pace by just the
right amount.
</item 9>

<item 10>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
But 40 Wall Street has its own unique problems. Tenant departures meant occupancy slid to 77 per cent in June, down from 98 per cent in 2015. Its net operating
income is only 61 per cent of what was projected by underwriters. Loading “There’s not a lot of leasing velocity down there,” said Albert Sultan, a broker at
Kassin Sabbagh Realty. The financial district used to offer a lower cost for tenants, but with Midtown rents declining, “there’s no reason to go downtown.” The
property is also subject to a ground lease that resets in 2032 at either 6 per cent of the land value or 85 per cent of its rent in the prior year. Based on
current rent, that means the ground lease would jump to $US30 million from about $US2.6 million now.
</item 10>

<item 11>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Olivier Blanchard, a former IMF chief economist, has long argued in favor of raising the inflation target, and former European Central Bank Vice President Vitor
Constancio has also embraced the idea. But it’s a controversial view and only possible from a position of credibility, which means central banks would likely
have to get inflation back to 2% first.

“It would be a mistake of the first order to think you can change a goal you have set if you can’t achieve it,” according to Bundesbank President Joachim Nagel.

Global trends suggest inflation will be stronger than in the past, with former Bank of England Governor Mark Carney among those saying rates won’t return to
pre-pandemic lows.

One lesson Gita Gopinath, the IMF’s No. 2 official, draws from the latest inflation episode is that policymakers mustn’t assume that looking through supply
shocks — as text books suggest — is the optimal response.
</item 11>

<item 12>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
ETF provider Betashares, which manages $30 billion in funds, reached an agreement to acquire Bendigo and Adelaide Bank’s superannuation business, in its first
venture into the superannuation sector. Betashares said it was part of a longer-term strategy to expand the business into the broader financial sector. Shares
in Bendigo increased 0.6 per cent on the news. REITS (up 0.4 per cent) was the strongest sector on the index as Goodman added 0.5 per cent and Dexus climbed 2.8
per cent. The laggards Casino operator Star Entertainment Group’s shares hit an all-time low of 60¢ after it raised $565 million. They closed the session 16 per
cent weaker at 63¢. Star, which raised $800 million in February, has had to return to the market for fresh funding and is hoping to raise $750 million at a
share price of 60¢ a share.
</item 12>

<item 13>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Ford Motors slumped 1.9 per cent after the United Auto Workers union escalated its walkout against Detroit automakers. In a surprise move, 8700 workers left
their jobs at a Ford truck plant in Louisville, Kentucky. In stock markets abroad, indexes were mixed in Europe after rising sharply in much of Asia.

Tweet of the day Quote of the day “With the continued growth of distributed work, Loom helps bridge the collaboration gaps for dispersed teams in deeply human
ways,” said Loom co-founder and billionaire Mike Cannon-Brookes as Atlassian announced it was acquiring former tech unicorn Loom, a US-based pioneer in
enterprise video chat and screen recording. You may have missed
</item 13>

<item 14>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Throughout September, in Tokyo, New York, Hong Kong and London, the world’s biggest investment banks hosted a series of conferences for the world’s biggest
investors, all gravitating around the same theme: is Japan really back?

And, according to both the attendees and hosts of those events — which were run by JPMorgan, Goldman Sachs, Bank of America, Mizuho, BNP Paribas, and Jefferies
— the answer is now clearer than it has been in a very long time. Japan is back, but global investors may take a little more persuasion that it is truly back
for the duration.

So, this week, prime minister Fumio Kishida will make his strongest ever push for a bet on Asia’s biggest advanced economy — as BlackRock founder Larry Fink,
and other heads of global funds, gather in Tokyo for a series of events to attract investment.
</item 14>

<item 15>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“I would urge you to evaluate what we are doing in my country, look at the underlying strength of our economy and our plans for the future and then invest in
Japan,” Kishida said in a recent speech at the Economic Club of New York.

On the face of it, the Tokyo conference guests have been told by various experts that the stars do appear aligned in Japan’s favour, both on its own merits and
in comparison with what is happening around it. But, because so many global funds have diverted expertise away from Japan in recent years, they are concerned
that they need to know more, as a matter of urgency.

“There is heightened interest in Japan after what has been, for many, a long period of not being focused on that country,” says Paul Hitchens, the head of the
Japan research department at Jefferies, which hosted a major investor conference in Hong Kong.
</item 15>

<item 16>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Even so: one wonders if some new changes at HSF, announced today, are related to this ‘continuation vote’ – and whether or not they amount to concessions to
help keep HSF investors happy and confident in the firm’s long-term prospects… and voting ‘yes, let’s keep going’.

The shareholder frustration: NAV vs. public market

First, a bit of important context.

The fundamental revenue performance of HSF doesn’t seem to be under scrutiny from shareholders: the company posted its best-ever revenues in the 12 months to
end of March, while streaming ‘pro-forma’ revenue was up 14.8% YoY in calendar 2022.

Where HSF shareholders – as well as its management – are finding frustration is in the current gulf between the firm’s ‘operative Net Asset Value’ (‘operative
NAV’) and its value on the public market (i.e. its share price).
</item 16>

<item 17>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Sycamore said the property sector and interest rate sensitive technology sector reclaimed some lost ground, while banks advanced on the back of the pullback in
yields.

However, iron ore miners were shackled by concerns around the Chinese economy, Sycamore said, as the market braced for China’s consumer purchasing index report
next week, which is expected to show inflation remaining morbid. The local rebound came after Wall Street rose after getting some relief from relaxing bond
yields and falling oil prices. Loading The S&P 500 climbed 0.8 per cent to claw back more than half its sharp tumble from a day earlier, which sent it to a
four-month low. The Dow Jones rose 0.4 per cent, a day after erasing the last of its gains for the year so far. The Nasdaq composite led the market with a gain
of 1.4 per cent. Stocks have struggled under the weight of soaring Treasury yields in the bond market.
</item 17>

<item 18>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 18>

<item 19>
Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers (Fortune,
2023-12-06T21:36:17+00:00)
“No, Senator,” Fraser answered. “We work hard to protect our customers and make sure that they don’t fall into overdraft.”

Menendez used this line of questioning to encourage Scharff, Dimon, and Moynihan to learn from Fraser and Citigroup. “I suggest you call Mr. Fraser after this
hearing to figure out how you can still be able to eliminate the fees in their entirety and still run a profitable bank,” Menendez urged Scharff, Moynihan, and
Dimon.

The other banks have made some steps to ease overdraft fees but haven’t done away with them completely. In 2021, JPMorgan Chase eliminated fees for overdrafts
that were under $50. The new policy still led to an increase in revenue from overdraft fees, which rose 3% in 2022, a year after the policy was implemented,
according to Bloomberg.
</item 19>

<item 20>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
The laggards Meanwhile, healthcare companies (down 0.1 per cent) fell despite CSL (down 0.4 per cent) delivering broadly in line with expectations. Pro Medicus
(down 0.2 per cent) and Ramsay Health Care (down 0.5 per cent) also fell. Bendigo and Adelaide Bank (down 1.3 per cent) and Westpac (down 0.2 per cent) were
among the biggest large-cap decliners, while Bank of Queensland shed 6.9 per cent after missing consensus expectations. Lithium miner Liontown (down 3.3 per
cent), TPG Telecom (down 1.3 per cent) and gold miner Northern Star (down 1 per cent) were also among the biggest large-cap decliners.

The lowdown TMS Capital portfolio manager Ben Clark said the Australian sharemarket was stronger on the back of a calming bond market and positive lead from
Wall Street, as well as positive news out of China.
</item 20>

<item 21>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.
</item 21>

<item 22>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
This can help the housing market retain some of its momentum without falling apart.”

Bank of America economists believe that low inventory of existing homes for sale combined with solid home sales to millennials could help put a “floor beneath
home prices,” but that doesn’t mean there won’t be some near-term pain due to higher mortgage rates.

“With rates likely staying higher for longer, we are cautious of potential turbulence ahead,” the group warned. Bank of America has previously forecast 0% home
price growth and falling home sales for the full year 2023, but it didn’t offer a new prediction in its latest note.

Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve. “At that stage, we should see a more stable and
healthy housing market,” Park and her team wrote. “Until then, hang tight, it may be a bumpy ride.”

This article has been updated to correct a mistaken term used to refer to Jeseo Park.
</item 22>

<item 23>
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
</item 23>

<item 24>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.
</item 24>

<item 25>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The laggards On the losing end, energy companies (down one per cent) weighed down the index. Woodside (down 0.8 per cent), Santos (down 0.4 per cent) and Ampol
(down 1.2 per cent) dropped, along with coal miners Yancoal (down 1.7 per cent) and Whitehaven (down 2 per cent). Lynas Rare Earths (down 3.1 per cent) was the
biggest large-cap decliner, while industrials companies (down 0.2 per cent) including Computershare (down 2.9 per cent), Seven West Holdings (down 1.9 per cent)
and Worley (down 1.3 per cent) were also weaker. The lowdown

TMS Capital portfolio manager Ben Clark said the big news for markets on Monday was the trading update from Westpac, which he said led a resurgence in the
banking sector’s performance.
</item 25>

<item 26>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
There are a few clues in HSF’s latest public statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being
“majority-owned by funds managed and/or advised by Blackstone“.

In addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount rate’ used by
Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).

Citrin Cooperman continues to stick to an 8.5% discount rate in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to
move this 8.5% figure upwards to, say, 9% or 9.5%.

If Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.
</item 26>

<item 27>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
</item 27>

<item 28>
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
</item 28>

<item 29>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
It comes as Chinese President Xi Jinping stepped up support for the world’s second-biggest economy and major commodity trading partner for Australia, issuing
additional sovereign debt, raising the budget deficit ratio and even making an unprecedented visit to the central bank. Loading Elsewhere, on Wall Street, the
benchmark S&P 500 index closed 0.7 per cent higher ahead of tech giants Microsoft and Google’s parent company Alphabet reporting their results after the close.
The Dow Jones Industrial Average added 0.6 per cent and the tech-heavy Nasdaq 100 gained 0.9 per cent. Shares in Microsoft climbed 5 per cent in after-hours
trading after the tech company said its sales jumped 13 per cent to $US56.5 billion ($88.9 billion) and profits climbed 27 per cent to $US22.3 billion in the
three months through September.

Meanwhile, shares in Alphabet fell 6 per cent after hours, after the Google parent delivered a mixed result.
</item 29>

<item 30>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Meanwhile, healthcare heavyweight CSL shed 1.4 per cent, weighing down the healthcare sector and insurance companies IAG (down 2.6 per cent) and Suncorp (down 2
per cent) gave back some of their gains from Tuesday. Gold miners Newcrest (down 2.1 per cent) and Evolution (down 3.5 per cent) were also among the biggest
large-cap decliners after the spot gold price dropped 0.9 per cent overnight. Information technology (down 1.1 per cent) was the weakest sector on the local
bourse with WiseTech losing 1.4 per cent. The lowdown

Novus Capital senior client adviser Gary Glover said the Australian sharemarket was surprisingly resilient following a negative lead from Wall Street and the
latest inflation data, with markets starting to wake up to the fact that interest rates could stay higher for longer. “Considering the damage overnight in the
US, Australian markets held on pretty well,” he said.
</item 30>

<item 31>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Brent crude, the international standard, rose US18¢ to $US86 a barrel. After jumping since the summer and then regressing a couple of weeks ago, crude prices
have been shaky following the latest fighting in Gaza. The worry is the violence could lead to disruptions in the supply of petroleum. Crude prices have been
shaky following the latest fighting in Gaza. Credit: Bloomberg Higher oil prices add costs across the economy, and airlines are particularly affected because
fuel is one of their biggest expenses.

Delta Air Lines fell 2.3 per cent despite reporting stronger profit for the summer than analysts expected. It also reported encouraging trends for bookings
going into the holiday season. The result heads a reporting season for S&P 500 companies that could mark a return to profit growth after three straight quarters
of declines. Several financial giants will report on Friday, including Citigroup, JPMorgan Chase and Wells Fargo, along with UnitedHealth Group.
</item 31>

<item 32>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Housing market pessimists have been sounding the alarm for years about a pending crash in the U.S. residential real estate market. Even before the Federal
Reserve began hiking interest rates to fight inflation last year, pushing mortgage rates to a 23-year high this month, years of surging home prices left some
experts warning that the housing market was a massive bubble ready to pop.

However, despite a widely bearish outlook in the industry, most real estate veterans have avoided arguing that home prices will drop like they did during the
2008 crash that kicked off the Global Financial Crisis (GFC). And so far, that’s been a wise decision.

It took until June 2022 for national home prices to peak, even as mortgage rates soared and purchase applications plummeted amid the Fed’s first interest rate
hikes. And while prices fell more than 5% from that peak by January of this year, they returned to a record high in July.
</item 32>

<item 33>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
In markets abroad, stock indexes were mixed across much of Europe. Tweet of the day

Quote of the day “Last year, Christmas spending was bolstered by a record freedom spending phenomenon with delayed overseas travel, whereas this year shoppers
are expected to be much more conservative with their budgets,” said Australian Retailers Association chief Paul Zahra as cost-of-living pressures put a lid on
spending, with shoppers tipped to cut back on hospitality, home goods and clothes purchases. You may have missed Qantas’ freight division has suffered a
catastrophic system failure that left dead bodies, live animals and perishable food unable to be collected from airports.
</item 33>

<item 34>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Bitcoin briefly topped $US35,000, while Treasury 10-year yields edged lower, following Monday’s intense volatility. Investors looking to the earnings season for
a dose of good news were hanging their hopes on big tech. The five biggest companies in the S&P 500 — Apple, Microsoft, Alphabet, Amazon and AI chipmaker Nvidia
— account for about a quarter of the benchmark’s market capitalisation. Their earnings are projected to jump 34 per cent from a year earlier on average,
according to analyst estimates compiled by Bloomberg Intelligence.

“As these big tech stocks go, so does the overall market,” David Trainer, chief executive officer of New Constructs, said before Microsoft and Alphabet’s
results. “Strong big tech earnings may just be what’s needed to end the stock market correction that started in late July. If big tech companies blow their
numbers out of the water and provide strong guidance for future earnings, then we could see the stock market rally strongly through the end of the year.”
</item 34>

<item 35>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Iron & steel have also witnessed accelerated long-term loan disbursements in the past six months. Infrastructure is witnessing an increase in credit flow with
67 per cent of the respondents indicating an increase in long-term loans as against 57 per cent in the previous round.

The survey suggests that the outlook on expectation on growth of non-food industry credit over next six months is optimistic with 42 per cent of the
participating banks expecting non-food industry credit growth to be above 12 per cent (as compared to 36 per cent in the previous round).

Rajan Jain, Head, Credit Research, SBI Caps, said in a note that whether the yield curve dynamics mean the US is moving to a structurally higher-rate economy
vs. post GFC normal remains a moot question, with important implications for investment demand and system risk.

These movements have had a cascading effect on global economies, which have seen routs in their respective bond markets, besides a fresh round of currency
depreciation.
</item 35>

<item 36>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Park and her team noted that there are no “noticeable signs” of excess housing development today like there were back then, and households aren’t nearly as
burdened by mortgage debt. Household mortgage debt represented roughly 65% of U.S. consumers’ disposable income in the second quarter, compared to 100% before
the GFC. And the ratio of Americans’ mortgage debt to their real estate assets—also called loan-to-value—was just 27% in the second quarter, compared to over
40% in 2008 and roughly 50% in 2010, Bank of America’s data shows.

New legislation was also enacted since the GFC to help prevent worst-case scenarios in the housing market. One of the most obvious effects of these new laws is
that there are far fewer risky adjustable-rate mortgages today.
</item 36>

<item 37>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 37>

<item 38>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
The Australian dollar was US64.20¢ at the close as the US dollar fell for a fifth straight day, its longest losing streak since July. ‘Everybody has one eye on
the Middle East conflict and one eye on what’s happening with bond yields.’ John Praveen, managing director at Paleo Leon The 10-year Treasury yield came off
its 16-year peak, on track for its steepest single-day drop since May, as trading resumed in the US bond market, which had been closed for a holiday on Monday.
Ten-year yields tumbled as much as 18 basis points to 4.62 per cent. Fed swaps currently show about a 60 per cent chance the Fed will stay on hold in December,
compared with 60 per cent odds on another interest rate rise by then, just a week ago. But while oil prices eased after Monday’s rally, investors were still
closely monitoring developments the Middle East.

Israeli air strikes attacked Gaza on Tuesday, razing entire districts in the densely populated and impoverished enclave.
</item 38>

<item 39>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
In sharemarkets abroad, indexes were modestly higher in Europe after exchanges were closed across much of Asia. Tweek of the day Quote of the day
</item 39>

<item 40>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.
</item 40>

<item 41>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Concerns about the conflict spreading more broadly have eased amid growing calls within Israel to rethink a ground invasion of Gaza because of fears about the
fate of some 200 hostages held there, the danger of retaliation by Hezbollah and the risk of Israeli military casualties. Tweet of the day Quote of the day “I
think when you get to those decisions, you can then sit on them for a while… or you can say let’s just deal with it [and] that’s what we have done today,” said
Magellan chairman Andrew Formica as the group’s chief executive stepped down abruptly amid the company’s efforts to turn around its performance.
</item 41>

<item 42>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Rising rates have made already stretched big tech valuations look increasingly expensive, with the group remaining the most-crowded trade among fund managers,
according to Bank of America. That’s prompted some investors to pay up for protection against a sell-off in Alphabet and Microsoft — two of the handful of
heavyweights responsible for all of the S&P 500 Index’s advance this year. The pain in long-duration growth stocks, fuelled in recent weeks by a relentless
surge in Treasury yields, is finally on the verge of subsiding. The yield on 10-year Treasuries was little changed at 4.85 per cent overnight. Oil closed at the
lowest in more than a week amid signs the crude market’s tightness has slackened and the Israel-Hamas war will remain contained for the time being. West Texas
Intermediate extended Monday’s drop with a 2.2 per cent fall to $US83.59 a barrel, paring most of the gains made after Hamas’ attack on Israel on October 7.
</item 42>

<item 43>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The average 30-year fixed mortgage rate, the most common type of mortgage in the U.S., has soared from 3.8% in March 2022 to over 7.5% today as a result. This,
in turn, has slowed mortgage purchase applications and caused home sales to plummet, just as it did in the ’80s; but home prices, echoing the dynamic of that
era, have yet to collapse.

One of the key reasons for the resilience of the housing market in both of these periods is demographics. “Noticeably, demographics were favorable back then,
with baby boomers having entered the prime homebuying age,” Park and her team wrote Thursday, arguing millennials are in a similar position today. “Some sales
activity should be supported by millennials reaching the prime homebuying age, and single-family building permits have steadily held up. This can help the
housing market retain some of its momentum without falling apart.”
</item 43>

<item 44>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
</item 44>

<item 45>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
On the small-cap front, Melbourne-based biotech Dimerix more than doubled its share price to 16¢ on Thursday after the company revealed it had inked an
exclusive licensing deal to commercialise its treatment for a kidney disease called Focal Segmental Glomerulosclerosis in Europe, the UK, Switzerland, Canada,
Australia and New Zealand. The laggards Meanwhile, energy companies (down 0.9 per cent) were among the weakest on the index as the oil price slumped overnight.
Heavyweights Woodside and Santos shed 0.8 per cent and 0.4 per cent, respectively, and Ampol dropped 2.8 per cent. Miners were also weaker, with lithium
prospector IGO (down 4 per cent) and Mineral Resources (down 2.9 per cent) among the biggest large-cap advancers.
</item 45>

<item 46>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Plus, the raising of interest rates in an effort to combat inflation is strikingly similar to the ’80s. Back then, Fed Chair Paul Volcker fought inflation
through aggressive interest rate hikes, with the average 30-year fixed mortgage peaking at about 18% by late 1981. Sound familiar? Current Fed Chair Jerome
Powell has set the tone for the most aggressive global hiking of rates in the modern era. And in due course, this week, mortgage rates hit 8%, the highest
they’ve been in more than 20 years.

All things considered, home sales activity also plummeted from 1978 to 1982. Existing-home sales dropped 50% during that period, according to the Office of
Policy Development and Research.
</item 46>

<item 47>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 47>

<item 48>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 48>

<item 49>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
The mining sector, however, managed to finish in positive territory. Gold miner Newmont was down 1.25 per cent, Origin Energy fell 0.4 per cent, Mercury Nz
declined 0.3 per cent and Argo Investments traded 0.1 per cent lower. The lowdown Moomoo market strategist Jessica Amir said the local sharemarket outperformed
global equities on Wednesday because of the National Accounts figures, which pointed to the Reserve Bank potentially hitting pause on lifting interest rates.
“We know economic growth is slowing, Australian unemployment is likely to pick up next year and inflation is likely to fall,” Amir said. “The good news is that
all of these three things, the trifecta, is pointing towards the RBA stopping rising interest rates and then potentially moving to cut interest rates. This is
what the market has priced in today.”
</item 49>

<item 50>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
They also want elected officials to put in place policies that help deliver sustainable growth.

“A change in mindset needs to happen,” said Agustin Carstens, the former governor of the Bank of Mexico who’s now the general manager of the BIS. “Growth needs
to depend less on fiscal and monetary policy, it should depend more on structural policies.”
</item 50>

<item 51>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And he’s not hopeful we’ll see home prices drop anytime soon: “Absent any negative shocks to the broader economy that would either boost excess supply of homes
on the market or fuel an uptick in unemployment, we continue to expect home prices to rise at a slow pace.” In fact, he predicts we’ll see home prices rise by
1.8% by the end of the year, with a 3.5% increase by the end of 2024.

Looking at more recent housing inventory data makes the issue even more stark. Between September 2018 and September 2023, the average number of homes on the
market dropped a whopping 60% to fewer than 700,000 active listings, according to Realtor.com.

“Unlike the turn of the millennium, house prices today are rising alongside mortgage rates, primarily due to low inventory,” Sam Khater, Freddie Mac’s chief
economist, said in a statement released Sept. 29.
</item 51>

<item 52>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
The laggards Real estate investment trusts (REITS, down 2 per cent) dropped further following the latest inflation data, with Mirvac (down 3.7 per cent) and
Dexus (down 3.3 per cent) sliding lower. Dexus chief executive Darren Steinberg on Wednesday announced he would step down from the position next year after 11
years at the helm. At the annual meeting in Sydney on Wednesday, the remuneration report received a first strike after failing to pass the 25 per cent voting
threshold. Loading Consumer staples (down 1.4 per cent) were also weaker with Woolworths losing 2 per cent despite sales across the company rising 5.3 per cent
in the 14 weeks to October, as it vowed to pass on easing costs for protein and fresh produce amid rising mortgage and rental pressures. In other company news,
shares in embattled funds manager Magellan dropped 3.5 per cent after the abrupt departure of its chief executive, David George.
</item 52>

<item 53>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The lowdown

Hotter-than-expected inflation data and the prospect of more rate rises in the world’s biggest economy weighed on sentiment in the Australian sharemarket as
interest-rate sensitive sectors such as IT and consumer discretionary dragged the index lower on Friday. While commodity prices boosted iron ore players,
company specific news was behind some of the biggest swings. Resmed extended its losses after brokers at RBC downgraded the stock and gold miner Newcrest
stepped up after investors approved its merger with US gold mining giant Newmont. US stocks slipped on Thursday as the clamps tighten on Wall Street from rising
yields in the bond market. The S&P 500 was 0.7 per cent lower in afternoon trading and on track to break a four-day winning streak, its longest since August.
The Dow Jones Industrial Average was down 204 points, or 0.6 per cent, as of 3pm Eastern time, and the Nasdaq composite was 0.8 per cent lower.
</item 53>

<item 54>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
"As for cutting the interest rate, it’s too early to consider that. The current interest rate is at the right level for the state of the economy, and together
with the policy on exchange rates, it contributes to ensuring financial stability," Leiderman concludes.

Published by Globes, Israel business news - en.globes.co.il - on October 12, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 54>

<item 55>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
The geopolitics of US-China tensions, chip wars, decoupling and “friendshoring” seem, in particular, to be working significantly in Japan’s favour. Asia-focused
global investment is rotating out of China and into Tokyo stocks, real estate and other assets, just as large parts of corporate Japan are engaging more
actively with shareholders than at any time in the past.

China, after years of stunning growth and investor attention, is very rapidly losing that sheen. Its economy is growing more slowly and the expected
post-pandemic rebound is proving elusive. On top of that, Beijing’s policymaking — with its sudden crackdowns on entire industries, like video games and private
tutoring, has generated a series of nasty surprises — and the unshakeable fear that others could occur at any time.

“You are seeing more interest in Japan from investors, who were primarily invested in China, as China has slowed and some of its economic policies have been
confusing and opaque,” says Kirk Neureiter, president of Fidelity Management & Research Japan.
</item 55>

<item 56>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives. The unprecedented move to remove a speaker from the
position likely doesn’t change much in the short term, with funding for the US government set until November 17.

“That said, a leadership vacuum in the House raises the odds of a government shutdown when the current funding extension expires,” according to economists at
Goldman Sachs. A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past
shutdowns. On Wall Street, big tech stocks helped to support the market after leading it lower a day earlier. They tend to move more sharply with expectations
for rates because high-growth stocks are seen as some of the biggest victims of high yields. In markets abroad, stock indexes were mixed across much of Europe.
</item 56>

<item 57>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
It’s again near its highest level since 2007.

Loading Treasury bonds are seen as some of the safest investments possible, and when they pay higher yields, investors are less likely to pay high prices for
stocks and other riskier investments. That’s a big reason why the S&P 500 dropped 4.9 per cent in September to drag what had been a big gain for the year down
to 11.7 per cent. Treasury yields have been climbing sharply as Wall Street accepts a new normal where the Federal Reserve is likely to keep interest rates high
for longer. The Fed is trying to push still-high inflation down to its target, and its main tool of high interest rates does that by trying to slow the economy
and hurting prices for investments. The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut
interest rates next year by less than it earlier expected.
</item 57>

<item 58>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
</item 58>

<item 59>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 59>

<item 60>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
On the losing end of Wall Street were stocks of energy producers, hurt by the slide in oil’s price. Energy stocks in the S&P 500 fell 2 per cent as a group,
more than double the loss of any of the other 10 sectors that make up the index. ExxonMobil fell 1.6 per cent, and Schlumberger dropped 4.3 per cent. Energy
stocks, though, remain the market’s standout performers since the summer. Shares of Ford and General Motors slipped after the United Auto Workers said it would
expand its limited strike to include another facility for each. Ford fell 1.1 per cent, and GM dipped 0.6 per cent.

All told, the S&P 500 slipped 11.65 points to 4288.05. The Dow dropped 158.84 to 33,507.50, and the Nasdaq added 18.05 to 13,219.32.
</item 60>

<item 61>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 61>

<item 62>
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
</item 62>

<item 63>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
“Deployment of capital to China has been a bit more risky in terms of eventual outcome of investment decisions, given that lack of clarity.” But interest in
Japan has not only come about because of the push away from China, he adds.

Japanese companies are increasingly aligning themselves with the interests of shareholders in a way that once seemed unlikely. Part of that, he says, is
generational. Companies that replace their chief executive on five or seven year cycles now have leaders that reached their positions after the establishment of
Japan’s stewardship code (in 2014) and corporate governance code (2015). Many have a very different view of the role of a chief executive than their
predecessors did.

“I think that Japan, in terms of the unifying message that companies and the Tokyo Stock Exchange are sending out, now gives a lot more confidence about where
they are deploying capital,” says Neureiter. “So people looking at the region are finding Japan incrementally more attractive.
</item 63>

<item 64>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.
</item 64>

<item 65>
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
</item 65>

<item 66>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“Everybody has one eye on the Middle East conflict and one eye on what’s happening with bond yields. The decline in bond yields is the key driver today,” said
John Praveen, managing director & co-chief investment officer at Paleo Leon. Loading While the Fed’s dovish comments were helping stocks on Tuesday and
investors were being sanguine about the Middle East, Praveen said that view could change if, for example, the fighting spread to other countries in the region.
“If tensions escalate bond yields might decline further because they’re a safe haven, but equities would sell off in that instance because of increased
uncertainty and risk aversion,” he said.

The Dow Jones Industrial Average rose 0.4 per cent, to 33,739.30, the S&P 500 gained 0.5 per cent and the Nasdaq Composite added 0.6 per cent. All 11 major S&P
500 sectors were advancing with materials and consumer staples leading gains, while technology was the slowest gainer.
</item 66>

<item 67>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 67>

<item 68>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
That was later than some analysts expected. Tweet of the day Quote of the day IAG chief executive Nick Hawkins said the behaviour of Peter Horton, the insurer’s
group general counsel and company secretary, didn’t meet expectations set out in its employee code of ethics and conduct. “This includes the importance of being
inclusive and respectful, and we will hold people to account if they fail to meet these expectations,” Hawkins said.
</item 68>

<item 69>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Loading Perpetual climbed 6.3 per cent after it announced a strategic review of its three-pronged corporate structure that might see it demerge its corporate
trust and wealth management businesses and create a more focused asset manager. Shares in Endeavour lifted 2.4 per cent after it forecast earnings before
interest and tax in its hotels division would grow by more than $150 million over the next five years. The laggards Utilities and energy stocks were the weakest
performers on the local bourse, rising 0.6 per cent and 0.8 per cent, respectively.

Only five large-cap stocks declined, with Evolution Mining sinking a mammoth 13 per cent after it raised $525 million selling new shares to institutional
investors at $3.80 a share. The mining sector, however, managed to finish in positive territory.
</item 69>

<item 70>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
The National Australia Bank has forecast another rate hike to 4.35 per cent in November. Elsewhere, Wall Street closed out its worst month and quarter of the
year with more losses on Friday. The S&P 500 slipped 0.3 per cent and the majority of stocks within the index sank. The Dow Jones fell 0.5 per cent and the
Nasdaq composite edged 0.1 per cent higher. The Australian sharemarket is set to retreat, with futures pointing to a fall of 38 points, or 0.5 per cent, at the
open. Solid gains for stocks early on faded as pressure built from within the bond market. After easing earlier in the day on encouraging signals about
inflation, Treasury yields got back to rising as the day progressed. The yield on the 10-year Treasury yield returned to 4.58 per cent, where it was late
Thursday, after dipping to 4.52 per cent. It’s again near its highest level since 2007.
</item 70>

<item 71>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The laggards Meanwhile, Harvey Norman (down 4.4 per cent) dropped as it went ex-dividend, weighing on the consumer discretionary sector (down 1 per cent).
Wesfarmers (down 0.2 per cent), Lottery Corporation (down 3.6 per cent) and IDP Education (down 1.3 per cent) also fell. Gold miners Evolution (down 3.4 per
cent) and Northern Star (down 2.6 per cent), along with NEXTDC (down 2.9 per cent) and REA Group (down 2.7 per cent) were among the biggest large-cap decliners.
Healthcare player Resmed (down 2.8 per cent) and lithium miner Liontown Resources (down 2.8 per cent) both bucked the trend among their respective sectors.
</item 71>

<item 72>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
These amendments would give HSM a 12-month notice period as HSF’s investment advisor should the IAA be terminated at any point following the upcoming 2023
‘continuation vote’.

Another clause in the new IAA: If HSF’s share price stands at an average discount to ‘operative NAV’ of 10% or more across the month of January 2025 (with that
NAV determined at the time of publication of HSF’s interim report for the period to September 30, 2024), the HSF board says it intends to serve notice to
terminate the IAA with HSM.

However, HSF’s board has also given itself the freedom to un-terminate the IAA during the entire notice period that would follow (e.g. if HSF’s board serves
12-month notice on HSM at the end of January 2025, it would have until the end of January 2026 to withdraw its termination decision).
</item 72>

<item 73>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Japanese prime minister Fumio Kishida will appear this week at a series of events aimed at attracting foreign investment © AFP via Getty Images

There are distinctive catalysts for that, he adds — noting a recent push by the Tokyo Stock Exchange for companies to improve their capital efficiency, and the
high profile decision of Warren Buffett to invest heavily in five of Japan’s largest trading houses.

Hitchens says: “Is there some healthy scepticism? Yes. However, global investment firms know that they need to invest and build their offerings in Japan to meet
the growing appetite, and that they need to pay attention to an equity market at a 33-year high.”

Topix, Japan’s main stock index, has risen more than 25 per cent this year, outperforming almost all the world’s developed market indices and, as some
conference attendees admit, creating a “fear of missing out” on a rally that may have significantly further to go.
</item 73>

<item 74>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.
</item 74>

<item 75>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Dow Jones Industrial Average gained 222.24, or 0.7 per cent, to 34,061.32, and the Nasdaq composite jumped 184.09, or 1.4 per cent, to 13,478.28. Stocks
surged through the week on rising hopes that the Federal Reserve is finally done with its market-crunching hikes to interest rates, in order to get inflation
under control. A report on Friday underscored that pressure is easing on inflation after it showed employers hired fewer workers last month than economists
expected. It’s a stunning turnaround from just a week ago, when Wall Street was reeling after the S&P 500 had fallen 10 per cent below its high point for the
year. That sent Wall Street’s main index into what investors call a “correction.” Since then strong profit reports helped drive some stocks to towering gains.
Generac, a maker of back-up generators, soared nearly 28 per cent for its best week since its stock began trading in 2010.
</item 75>

<item 76>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The rise in the consumer price index peaked at around 14% in 1980 before then–Fed Chair Paul Volcker’s hawkish policies sent mortgage rates to 18% in a year’s
time, cooling inflation, but sparking a recession. This caused a serious downturn in the housing market in which home sales and building levels cratered.
However, national home prices actually remained stable.

At the start of Volcker’s term as Fed chair in August 1979, the median U.S. home sales price was $64,700. And even after a near doubling of mortgage rates, that
figure rose to $69,400 by the second quarter of 1981.

Over the past 18 months, the Fed’s current chair, Jerome Powell, has been following a very similar game plan to Volcker’s, raising interest rates aggressively
to quash inflation.
</item 76>

<item 77>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Google, which is also seeking to capitalise on surging demand for AI technology, saw revenue climb 11 per cent to $US76.7 billion, with net income of $US19.7
billion thanks to a recovery in digital advertising revenue. However, its closely watched cloud division, which houses its data storage business and many of its
AI efforts, missed Wall Street expectations. “Investors were disappointed by the relatively weak performance at its Google cloud platform, which is at risk of
falling further behind” competitors’ offerings, Jesse Cohen, a senior analyst at Investing.com, wrote in an email. Loading During trading hours, the S&P 500
slipped off session highs, led by losses in energy shares as oil slid below $US84 a barrel, but once again the index found support above the key 4200 mark.
Verizon Communications, 3M and General Electric climbed on bullish forecasts. Facebook and Instagram owner Meta dropped after being sued by California and dozen
of other US states over harmful youth marketing claims.
</item 77>

<item 78>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Billionaire investor Paul Tudor Jones told CNBC the current geopolitical environment was the “most threatening and challenging” he’d ever seen following Hamas’s
attack on Israel over the weekend and predicted the US will enter into a recession early next year. The shekel regained its footing as the central bank duelled
short sellers to contain the market fallout from Israel’s conflict with militant group Hamas. European gas prices jumped, with Finland on high alert as it
suspects a gas pipeline leak in the Baltic Sea was caused by a deliberate act of destruction, fuelling concerns about the safety of Europe’s energy
infrastructure. Base metals sank amid fresh fears over property turmoil in China. Tweet of the day

Quote of the day “This is as strong a level of confidence in the market that we have seen since before COVID,” said The Leasing Agency group managing director
Terence Yeh as the return of international students and a growing number of apartments in inner-city Melbourne provides a shot in the arm for CBD hospitality
businesses.
</item 78>

<item 79>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Tweet of the day Quote of the day “The Senate committees have the power to summons witnesses within Australia but have no enforceable powers for witnesses who
are overseas,” said Senator Bridget McKenzie as former Qantas boss Alan Joyce chose not to front the Senate select committee into the federal government’s
decision to reject extra flights from Qatar Airways due to “personal commitments”.
</item 79>

<item 80>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Large global funds are not the only ones to have had their interest in Japan piqued, says Keith Truelove, head of global markets at UBS Securities Japan.

“There is still a lot of dry powder on the sidelines,” he observes. “There is a lot of interest in Japan from family offices and from high net worth individual
investors, who value wealth preservation more highly than institutional investors — it’s really taking off now.”

“They took their time to decide if it was safe to invest in Japan again, because they need to get confidence in the idea that the Japan story will be more
sustainable this time,” Truelove adds. He notes that Japan has offered investors a number of false-dawns over the years, and the bar for deciding that ‘this
time is different’ remains high.

One of the most significant changes, say asset managers, is the fundamental shift in the way global funds now look at China.
</item 80>

<item 81>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
“Considering the damage overnight in the US, Australian markets held on pretty well,” he said. “I thought it would be a bigger down day across the board.”
Glover said the market was volatile but quite range-bound, similar to previous periods of high inflation in the 1940s and 1970s. Elsewhere, Wall Street’s ugly
September got even worse on Tuesday, as a sharp drop for stocks brought them back to where they were in June. The S&P 500 tumbled 1.5 per cent for its fifth
loss in the last six days. The Dow Jones dropped 1.1 per cent, and the Nasdaq composite lost 1.6 per cent.

Loading September has brought a loss of 5.2 per cent so far for the S&P 500, putting it on track to be the worst month of the year by far, as the realisation
sets in that the Federal Reserve will indeed keep interest rates high for a long time.
</item 81>

<item 82>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Stocks have struggled under the weight of soaring Treasury yields in the bond market. High yields undercut stock prices by pulling investment dollars away from
stocks and into bonds. They also crimp corporate profits by making borrowing more expensive.

The yield on the 10-year Treasury, which is the centrepiece of the bond market, pulled back from its highest level since 2007, down to 4.73 per cent from 4.80
per cent late on Tuesday. Shorter- and longer-term yields also eased to allow more oxygen for the stock market. Yields fell following a couple of reports
indicating a slowing US economy. The first suggested hiring by employers outside the government was much weaker last month than expected. On Wall Street, that’s
currently good news because a cooling job market could mean less upward pressure on inflation. That in turn could convince the Federal Reserve to take it easier
on interest rates.
</item 82>

<item 83>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Brent crude, the international standard, also weakened. The resumption of US student-loan repayments, meanwhile, may funnel more dollars away from the spending
by consumers that has helped to keep the economy afloat. The latest monthly update on the US jobs market is due next week, with a couple of important reports on
inflation coming the following week. Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on
interest rates based on what incoming data say about the economy. The Fed’s next meeting on rates ends on November 1. On Wall Street, Nike jumped 6.7 per cent
after reporting better profit for the latest quarter than analysts expected. Strength overseas helped it make up for some declines in North America.

Loading Blue Apron soared 134.5 per cent after the meal kit company said it was being bought by Wonder Group for $US13 per share in cash in a deal valued at
$US103 million ($160.2 million).
</item 83>

<item 84>
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
</item 84>

<item 85>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
Alongside this, tourism, commerce, transport services, entertainment, and restaurants will be hurt. As in the Covid pandemic, the self-employed and small and
medium-size businesses may be hurt, particularly those that operate with high leverage."

On Monday, the Bank of Israel announced a dramatic plan to sell $30 billion of its foreign currency reserves to stabilize the shekel. The shekel has in fact
recovered slightly, and is traded at around NIS 3.96/$.

What’s your view of the Bank of Israel’s move in the foreign exchange market?

Leiderman: "It’s important to welcome the Bank of Israel’s decision to intervene, as necessary, in the foreign exchange market, in order to ensure the market’s
regular operation. The decision to extend the term of Amir Yaron as governor of the bank at least for the duration of the war, and I hope beyond that, is also a
stabilizing factor. The banking system is stable, and there is enough liquidity in the system.
</item 85>

<item 86>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Loading Wall Street has dealt with such shutdowns in the past, and stocks have historically been turbulent in the run-up to them, according to Lori Calvasina,
strategist at RBC Capital Markets. After looking at the seven shutdowns that lasted 10 days or more since the 1970s, she found the S&P 500 dropped an average of
roughly 10 per cent in the three months heading into them. But stocks managed to hold up rather well during the shutdowns, falling an average of just 0.3 per
cent, before rebounding meaningfully afterward. Wall Street is also contending with higher oil prices, shaky economies around the world, a strike by US
autoworkers that could put more upward pressure on inflation and a resumption of US student-loan repayments that could dent spending by households.

On Wall Street, the vast majority of stocks fell under such pressures, including 90 per cent of those within the S&P 500.
</item 86>

<item 87>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Big Tech stocks tend to be among the hardest hit by high rates, and they were the heaviest weights on the index. Apple fell 2.3 per cent and Microsoft lost 1.7
per cent. Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it. They accuse the
e-commerce behemoth of using its dominant position to inflate prices on other platforms, overcharge sellers and stifle competition. In China, concerns continued
over heavily indebted real estate developer Evergrande. The property market crisis there is dragging on China’s economic growth and raising worries about
financial instability. France’s CAC 40 fell 0.7 per cent, and Germany’s DAX lost 1 per cent.

Crude oil prices rose, adding to worries about inflation. A barrel of benchmark US crude climbed 71¢ to $US90.39. Brent crude, the international standard, added
67¢ to $US93.96 per barrel.
</item 87>

<item 88>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That will arrive on Friday. But “if Friday’s report also shows the labour market is cooling, stock investors may worry a little less about indefinitely higher
interest rates,” said Mike Loewengart, head of model portfolio construction at Morgan Stanley Global Investment Office. A second report on the economy said
growth in US services industries slowed in September by a touch more than economists expected.

It also offered some hints of sticky pressure on inflation, with prices paid by services companies rising last month at a similar rate as in August. Oil prices
tumbled on Wednesday to take some heat off inflation. Benchmark US crude fell $US5.01 to settle at $US84.22 per barrel for its worst drop in just over a year.
It’s been pulling back since, topping $US93 last week. Brent crude, the international standard, lost $US5.11 to $US85.81. Loading Prices for crude had been
generally charging higher from $US70 during the summer, following announcements of cuts to production by some oil-producing countries.
</item 88>

<item 89>
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
</item 89>

<item 90>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 90>

<item 91>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Friday’s economic data shows that not only was inflation a touch cooler than expected in August, so was growth in spending by US consumers. That can be a
positive for inflation because it means not as many dollars are pouring into purchases. That in turn could give companies less encouragement to try to raise
prices further. But it may also dent what’s been a big driver keeping the US economy out of a recession.

“It came to a boil during a hot summer, and the temperature is really starting to come down,” said Brian Jacobsen, chief economist at Annex Wealth Management,
of spending growth by US consumers. “Higher energy prices, student loan debt repayments and real disposable incomes that have been on a declining trajectory
since June doesn’t bode well.” Oil prices have jumped to their highest level in more than a year, which is pressuring the economy by raising fuel costs for
everyone. A barrel of US crude sank 92¢ Friday to settle at $US90.79, but it’s still up sharply from $US70 in June.
</item 91>

<item 92>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
But a spread of the crisis to the whole Middle East can risk trade through this route, which accounts for 20 per cent of global oil trade passage. If Iran is
embroiled into this conflict, then trade with landlocked Central Asia, which is a big agricultural producer, raw materials and minerals like copper, cotton,
uranium, gold, natural gas, gets disrupted. Central Asia also hosts a network of road, rail, and air routes that link China, Europe, and South Asia.

Impact on Indian markets

In the above scenario, supply disruption will lead to higher inflation and central bankers will have to adjust their policies and hold rates higher for longer
than anticipated. We may witness a scenario where rates are held higher till the year 2025. Higher crude oil bill and higher global rates will prompt the RBI to
also remain hawkish in its approach with regards to rates and liquidity measures. EBITDA margins of India Inc. may again get impacted on account of higher input
costs and higher freight and fuel costs.
</item 92>

<item 93>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
You may have missed Corporate leaders caught underpaying staff should expect zero sympathy from new Fair Work Ombudsman Anna Booth, who believes chief
executives and board directors are directly responsible for any failures to pay correct wages. With Bloomberg/Reuters
</item 93>

<item 94>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 94>

<item 95>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Adjustable-rate mortgages can lead to higher default rates when interest rates rise, but they now represent less than 5% of total purchase and refinance loans,
compared with over 35% at the peak of the pre-GFC housing cycle.

“We reiterate that we do not expect another housing crash like 2008,” Park and her team concluded after presenting this evidence.

Is it a repeat of the ‘turbulence’ of the 1980s?

According to Bank of America, today’s housing market looks a lot more like the early 1980s than it does 2008. Back then, just like today, home prices had boomed
for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
</item 95>

<item 96>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Tweet of the day

Quote of the day “Our view is that in the high likelihood there is another cash rate increase on Tuesday, that is going to be the nail in the coffin for
consumer confidence,” said CreditorWatch chief economist Anneke Thompson, amid warnings from some economists that another rate rise, as some market watchers are
predicting, will dampen spending intentions just in time for Christmas. You may have missed The men who built pubs and liquor juggernaut Endeavour Group into
the world’s largest poker machine operator have triggered a public spat over the future of a business that they say has lost its way.
</item 96>

<item 97>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.
</item 97>

<item 98>
U.S.-China relations are now more about crisis prevention (Cnbc | World Business News Leader, 2023-11-13T23:46:00+00:00)
Chinese Vice Premier He Lifeng also raised the issues during preparatory meetings with U.S. Treasury Secretary Janet Yellen in San Francisco on Nov. 10,
according to state media. "Aside from Taiwan, export controls are Beijing's top concern, but there is no political space in Washington roll back existing
controls," Gabriel Wildau, managing director at consulting firm Teneo, said in a note. "The immediate aftermath of the [Biden-Xi] meeting is likely to mark a
cyclical high point for bilateral relations," he said. "The key question is whether this high point extends into a plateau or whether political pressures
trigger a new cycle of deterioration," he said. "As previously discussed, the period since June has offered a window of opportunity to stabilize relations;
following the meeting, this window may close." Taiwan is set to hold its presidential election in January, and a more pro-independence winner could stir more of
Beijing's ire.
</item 98>

<item 99>
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
</item 99>

<item 100>
Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers (Fortune,
2023-12-06T21:36:17+00:00)
Research points to the fact that overdraft fees disproportionately affect Black and Hispanic customers. A 2021 study found that Hispanic Americans paid $3.1
billion in overdraft fees a year Black Americans were hit with $1.4 billion in total. Another survey found they paid double the amount of fees than White people
did.

Menendez commended Citibank’s Fraser for her company eliminating such fees in 2022 after government officials urged an end to the practice. Big banks like Citi
also face competitive pressures from fintech startups and smaller banks, like Ally Bank, that offer bank accounts without overdraft fees.

“Is it fair to say that Citi is still a profitable bank?” Menendez asked Fraser.

“Yes, Senator,” Fraser replied.

Menendez pressed her further, asking if their elimination led to wholesale changes at the bank.

“Did [removing] overdraft fees upend your business model,” Menendez continued.

“No, Senator,” Fraser answered.
</item 100>
=== END ITEM id=2915c95baca805ba ===
