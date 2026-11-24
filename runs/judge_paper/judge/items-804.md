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

This file has 4 items.

=== ITEM id=c9cd616d905f2bf9 ===
Gold sentences and facts:
Sentence 0: For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report.
  Fact 0.0: Investors around the world consider the Fed to matter more than other central banks.
  Fact 0.1: Foreign-exchange strategists at Bank of America say weak US data is the only game-changer for markets.
  Fact 0.2: The Bank of America foreign-exchange strategists wrote this in a BofA Global Research report.
Sentence 1: Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
  Fact 1.0: Postponements of such reports could complicate things for the Fed.
  Fact 1.1: The Fed has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.

Retrieved items:
<item 1>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
The Australian dollar was US64.20¢ at the close as the US dollar fell for a fifth straight day, its longest losing streak since July. ‘Everybody has one eye on
the Middle East conflict and one eye on what’s happening with bond yields.’ John Praveen, managing director at Paleo Leon The 10-year Treasury yield came off
its 16-year peak, on track for its steepest single-day drop since May, as trading resumed in the US bond market, which had been closed for a holiday on Monday.
Ten-year yields tumbled as much as 18 basis points to 4.62 per cent. Fed swaps currently show about a 60 per cent chance the Fed will stay on hold in December,
compared with 60 per cent odds on another interest rate rise by then, just a week ago. But while oil prices eased after Monday’s rally, investors were still
closely monitoring developments the Middle East.

Israeli air strikes attacked Gaza on Tuesday, razing entire districts in the densely populated and impoverished enclave.
</item 1>

<item 2>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.
</item 2>

<item 3>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 3>

<item 4>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 4>

<item 5>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Billionaire investor Paul Tudor Jones told CNBC the current geopolitical environment was the “most threatening and challenging” he’d ever seen following Hamas’s
attack on Israel over the weekend and predicted the US will enter into a recession early next year. The shekel regained its footing as the central bank duelled
short sellers to contain the market fallout from Israel’s conflict with militant group Hamas. European gas prices jumped, with Finland on high alert as it
suspects a gas pipeline leak in the Baltic Sea was caused by a deliberate act of destruction, fuelling concerns about the safety of Europe’s energy
infrastructure. Base metals sank amid fresh fears over property turmoil in China. Tweet of the day

Quote of the day “This is as strong a level of confidence in the market that we have seen since before COVID,” said The Leasing Agency group managing director
Terence Yeh as the return of international students and a growing number of apartments in inner-city Melbourne provides a shot in the arm for CBD hospitality
businesses.
</item 5>

<item 6>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The laggards On the losing end, energy companies (down one per cent) weighed down the index. Woodside (down 0.8 per cent), Santos (down 0.4 per cent) and Ampol
(down 1.2 per cent) dropped, along with coal miners Yancoal (down 1.7 per cent) and Whitehaven (down 2 per cent). Lynas Rare Earths (down 3.1 per cent) was the
biggest large-cap decliner, while industrials companies (down 0.2 per cent) including Computershare (down 2.9 per cent), Seven West Holdings (down 1.9 per cent)
and Worley (down 1.3 per cent) were also weaker. The lowdown

TMS Capital portfolio manager Ben Clark said the big news for markets on Monday was the trading update from Westpac, which he said led a resurgence in the
banking sector’s performance.
</item 6>

<item 7>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 7>

<item 8>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.
</item 8>

<item 9>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 9>

<item 10>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Meanwhile, healthcare heavyweight CSL shed 1.4 per cent, weighing down the healthcare sector and insurance companies IAG (down 2.6 per cent) and Suncorp (down 2
per cent) gave back some of their gains from Tuesday. Gold miners Newcrest (down 2.1 per cent) and Evolution (down 3.5 per cent) were also among the biggest
large-cap decliners after the spot gold price dropped 0.9 per cent overnight. Information technology (down 1.1 per cent) was the weakest sector on the local
bourse with WiseTech losing 1.4 per cent. The lowdown

Novus Capital senior client adviser Gary Glover said the Australian sharemarket was surprisingly resilient following a negative lead from Wall Street and the
latest inflation data, with markets starting to wake up to the fact that interest rates could stay higher for longer. “Considering the damage overnight in the
US, Australian markets held on pretty well,” he said.
</item 10>

<item 11>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
</item 11>

<item 12>
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
</item 12>

<item 13>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 13>

<item 14>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Stocks have struggled under the weight of soaring Treasury yields in the bond market. High yields undercut stock prices by pulling investment dollars away from
stocks and into bonds. They also crimp corporate profits by making borrowing more expensive.

The yield on the 10-year Treasury, which is the centrepiece of the bond market, pulled back from its highest level since 2007, down to 4.73 per cent from 4.80
per cent late on Tuesday. Shorter- and longer-term yields also eased to allow more oxygen for the stock market. Yields fell following a couple of reports
indicating a slowing US economy. The first suggested hiring by employers outside the government was much weaker last month than expected. On Wall Street, that’s
currently good news because a cooling job market could mean less upward pressure on inflation. That in turn could convince the Federal Reserve to take it easier
on interest rates.
</item 14>

<item 15>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
The National Australia Bank has forecast another rate hike to 4.35 per cent in November. Elsewhere, Wall Street closed out its worst month and quarter of the
year with more losses on Friday. The S&P 500 slipped 0.3 per cent and the majority of stocks within the index sank. The Dow Jones fell 0.5 per cent and the
Nasdaq composite edged 0.1 per cent higher. The Australian sharemarket is set to retreat, with futures pointing to a fall of 38 points, or 0.5 per cent, at the
open. Solid gains for stocks early on faded as pressure built from within the bond market. After easing earlier in the day on encouraging signals about
inflation, Treasury yields got back to rising as the day progressed. The yield on the 10-year Treasury yield returned to 4.58 per cent, where it was late
Thursday, after dipping to 4.52 per cent. It’s again near its highest level since 2007.
</item 15>

<item 16>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“Everybody has one eye on the Middle East conflict and one eye on what’s happening with bond yields. The decline in bond yields is the key driver today,” said
John Praveen, managing director & co-chief investment officer at Paleo Leon. Loading While the Fed’s dovish comments were helping stocks on Tuesday and
investors were being sanguine about the Middle East, Praveen said that view could change if, for example, the fighting spread to other countries in the region.
“If tensions escalate bond yields might decline further because they’re a safe haven, but equities would sell off in that instance because of increased
uncertainty and risk aversion,” he said.

The Dow Jones Industrial Average rose 0.4 per cent, to 33,739.30, the S&P 500 gained 0.5 per cent and the Nasdaq Composite added 0.6 per cent. All 11 major S&P
500 sectors were advancing with materials and consumer staples leading gains, while technology was the slowest gainer.
</item 16>

<item 17>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The lowdown

Hotter-than-expected inflation data and the prospect of more rate rises in the world’s biggest economy weighed on sentiment in the Australian sharemarket as
interest-rate sensitive sectors such as IT and consumer discretionary dragged the index lower on Friday. While commodity prices boosted iron ore players,
company specific news was behind some of the biggest swings. Resmed extended its losses after brokers at RBC downgraded the stock and gold miner Newcrest
stepped up after investors approved its merger with US gold mining giant Newmont. US stocks slipped on Thursday as the clamps tighten on Wall Street from rising
yields in the bond market. The S&P 500 was 0.7 per cent lower in afternoon trading and on track to break a four-day winning streak, its longest since August.
The Dow Jones Industrial Average was down 204 points, or 0.6 per cent, as of 3pm Eastern time, and the Nasdaq composite was 0.8 per cent lower.
</item 17>

<item 18>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Tweet of the day

Quote of the day “Our view is that in the high likelihood there is another cash rate increase on Tuesday, that is going to be the nail in the coffin for
consumer confidence,” said CreditorWatch chief economist Anneke Thompson, amid warnings from some economists that another rate rise, as some market watchers are
predicting, will dampen spending intentions just in time for Christmas. You may have missed The men who built pubs and liquor juggernaut Endeavour Group into
the world’s largest poker machine operator have triggered a public spat over the future of a business that they say has lost its way.
</item 18>

<item 19>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 19>

<item 20>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
It’s again near its highest level since 2007.

Loading Treasury bonds are seen as some of the safest investments possible, and when they pay higher yields, investors are less likely to pay high prices for
stocks and other riskier investments. That’s a big reason why the S&P 500 dropped 4.9 per cent in September to drag what had been a big gain for the year down
to 11.7 per cent. Treasury yields have been climbing sharply as Wall Street accepts a new normal where the Federal Reserve is likely to keep interest rates high
for longer. The Fed is trying to push still-high inflation down to its target, and its main tool of high interest rates does that by trying to slow the economy
and hurting prices for investments. The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut
interest rates next year by less than it earlier expected.
</item 20>

<item 21>
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
</item 21>

<item 22>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.
</item 22>

<item 23>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.
</item 23>

<item 24>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 24>

<item 25>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Treasury Wine Estates (down 1.9 per cent) and Woolworths (down 1.2 per cent) dragged the consumer staples sector (down 1.1 per cent) lower. Pilbara Minerals
(down 2.1 per cent) and Qantas (down 1.5 per cent) were also among the biggest large-cap decliners. The lowdown Stronger iron ore prices helped to bolster the
local bourse on Monday but not enough to offset the prospect of higher-for-longer interest rates ahead of the Reserve Bank’s meeting on Tuesday. Iron ore
heavyweights advanced after the iron ore price increased 1.4 per cent overnight with lithium miner Liontown resisting the downward momentum of lithium prices,
which dropped to the lowest level in two years.

Most sectors traded lower but interest-rate sensitive sectors including healthcare and consumer companies were among the weakest amid expectations that the RBA
will keep rates on hold in October. The National Australia Bank has forecast another rate hike to 4.35 per cent in November.
</item 25>

<item 26>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That will arrive on Friday. But “if Friday’s report also shows the labour market is cooling, stock investors may worry a little less about indefinitely higher
interest rates,” said Mike Loewengart, head of model portfolio construction at Morgan Stanley Global Investment Office. A second report on the economy said
growth in US services industries slowed in September by a touch more than economists expected.

It also offered some hints of sticky pressure on inflation, with prices paid by services companies rising last month at a similar rate as in August. Oil prices
tumbled on Wednesday to take some heat off inflation. Benchmark US crude fell $US5.01 to settle at $US84.22 per barrel for its worst drop in just over a year.
It’s been pulling back since, topping $US93 last week. Brent crude, the international standard, lost $US5.11 to $US85.81. Loading Prices for crude had been
generally charging higher from $US70 during the summer, following announcements of cuts to production by some oil-producing countries.
</item 26>

<item 27>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
</item 27>

<item 28>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Rising rates have made already stretched big tech valuations look increasingly expensive, with the group remaining the most-crowded trade among fund managers,
according to Bank of America. That’s prompted some investors to pay up for protection against a sell-off in Alphabet and Microsoft — two of the handful of
heavyweights responsible for all of the S&P 500 Index’s advance this year. The pain in long-duration growth stocks, fuelled in recent weeks by a relentless
surge in Treasury yields, is finally on the verge of subsiding. The yield on 10-year Treasuries was little changed at 4.85 per cent overnight. Oil closed at the
lowest in more than a week amid signs the crude market’s tightness has slackened and the Israel-Hamas war will remain contained for the time being. West Texas
Intermediate extended Monday’s drop with a 2.2 per cent fall to $US83.59 a barrel, paring most of the gains made after Hamas’ attack on Israel on October 7.
</item 28>

<item 29>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.
</item 29>

<item 30>
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
</item 30>
=== END ITEM id=c9cd616d905f2bf9 ===

=== ITEM id=bd4a038571d1be63 ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.0: On October 12, 2023, Israel warned 1.1 million Gaza residents to leave for the southern region.
  Fact 0.1: The warned Gaza residents were in the northern section of the enclave.
  Fact 0.2: Israel issued the warning in advance of a potential ground invasion.
Sentence 1: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 1.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 1>

<item 2>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 2>

<item 3>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas is an excuse, new lands are wonderful



I definitely support Israel pic.twitter.com/AbLdeazQjR October 7, 2023

Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that
infiltrated its territory. "The second objective, at the same time, is to exact an immense price from the enemy, within the Gaza Strip as well," he said, adding
that the third objective was to reinforce other fronts so that nobody should mistakenly join this war. "We are at war. In war, one needs to be level-headed. I
call on all citizens of Israel to unite in order to achieve our highest goal – victory in the war," Netanyahu said.

Since this morning, the State of Israel has been at war. Our first objective is to clear out the hostile forces that infiltrated our territory and restore the
security and quiet to the communities that have been attacked.
</item 3>

<item 4>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
"Despite the clear threat posed by Gaza, the decision makers preferred to avoid a policy that would mean returning a large part of the country's population to
the reality of air raids and disrupted daily routines. The bottom line: the Iron Dome did not change the essential strategic situation; as firepower from the
Gaza Strip developed, so did the terrorist organizations’ ability to control daily life in Israel’s hinterland, hold it hostage, and restrain Israel from taking
proactive measures against them."

"Why didn't the relative success of the Iron Dome translate into the even more important strategic achievement of restoring Israel's freedom of action against
Hamas in the Strip?" asks Ortal rhetorically, and names several factors. "The first, like any other operational system, Iron Dome is also not perfect. The enemy
studies it and develops its own challenges against it, and the decision makers recognized the fact that the State of Israel is not hermetically protected.
Second, the enormous cost involved in deploying and operating [Iron Dome] batteries significantly impedes a policy based on continuous rounds of violence.
</item 4>

<item 5>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
There had been a second line of defense, observation balloons, but for unknown reasons, all three had stopped working some weeks before the attack, and repairs
were postponed until "after the holidays."

Israel’s drone warfare fleet, which is capable of jamming signals of a hostile drone, and the attack drone fleet, known for warfare against incendiary balloons,
also did not work for unknown reasons, and dozens of Hamas drones operated without interruption. The IDF's Iron Beam high energy laser (HEL) interception
system, which was developed for such cases, among other things, has not yet entered into operational activity after many years of development. All allowed the
Hamas terrorists to quickly reach the IDF bases along the Gaza border, disable the information and communications technology (ICT) systems, neutralizing the
ability to call reinforcements without arousing much suspicion at IDF Kirya headquarters in Tel Aviv.

Above all there was a sense of security and belief in technology and the sophisticated barrier’s ability to prevent any form of intrusion.
</item 5>

<item 6>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 6>

<item 7>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
India shipped goods around $8billion to Israel. Petroleum products account for 65 per cent of the exports, followed by gems and jewellery (15 per cent) and
electrical machinery (3 per cent). Trade can be adversely impacted if operations at the Israeli ports are disrupted. Fertiliser movement from Israel’s Port of
Ashdod accounts for 3 per cent of global potash supply. Any disruption on this front will also be a dampener.

But, what if the current hostilities turn into a full-scale conflict?

What supports this proposition is the fact that while there have been many conflicts between Israel and Hamas in the past decades, there have been none of this
intensity. With at least 1,200 Israelis killed, this has been the biggest single-day killing of Jews since Holocaust, and Israel opposition is also united in
retaliating. Israel has termed it a ‘War’, formed a war management cabinet with a key opposition member, and vowed to change the look of the ‘Middle East’.
</item 7>

<item 8>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
They told themselves, we have B-52 bombers, there's no way we can't win."

Technological sophistication and the massive investment in intelligence means have greatly increased the IDF's visual intelligence gathering capabilities,
according to former IDF Intelligence Directorate head General (ret.) Aharon Ze'evi-Farkash. "I'm sure it doesn't come at the expense of human sources, but as
time goes by, gathering visual intelligence is easier than human sources and they bring better results. It's hard to bring in a Palestinian 'Ashraf Marwan,' the
agent who passed the information to Israel before the Egyptian attack in 1973. It’s a problem because the human source is essential to provide an interpretation
of the rest of the information that’s received, and to make decisions. Golda knew who Marwan was and expected to hear what he would say about the Egyptian
deployment in the south." At the same time, according to "The New York Times" a year before the attack, the IDF stopped monitoring Hamas' radio communications
last year.
</item 8>

<item 9>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
The US had put stock in Israeli intelligence regarding Hamas in recent years and hadn’t monitored it independently.

The human factor

The technological concept is, of course, not limited to the border fence, or to intelligence, and is not just a product of Operation Protective Edge. For years,
the IDF has been praising the ethos of a small and smart army, implicitly if not explicitly. In recent years, it has publicized countless technological projects
and present itself as a high-tech army: establishing a technological division, called Shiloh, to coordinate the development of technologies for all branches of
the army, and with the Directorate of Defense Research & Development (IMOD DDR&D or MAFAT).
</item 9>

<item 10>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Truly disturbing acts of terror directed at civilians".

Some of the images coming out of Sderot over the last few hours are reaching degrees of horror that I dont really think I can use words to properly describe.
Truly disturbing acts of terror directed at civilians. — OSINTtechnical (@Osinttechnical) October 7, 2023

Commenting on the coordinated attack, Rania Khalek, a journalist, said that the strategic planning that must have gone into this against "one of the most
high-tech occupation and surveillance systems in the world" will be studied by many. "The Israelis must be totally losing it right now."
</item 10>

<item 11>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 11>

<item 12>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Thirdly, as long as the rockets are intercepted in the skies over Israel, and not the skies over Gaza, the Israeli home front is forced to experience sirens,
security rooms and shelters, and disrupted routine living. In other words, even the complete success of the Iron Dome left daily life on the home front in the
hands of the terrorist organizations."

In 2018, Ortal wrote about "the barrier" that, "it would be wrong to assume it will turn out to be more immune than its predecessors. The longer as we delay
cutting the direct connection between the Hamas rockets and those of our other enemies, who make extensive use of this method of operation, and the idea of
attacking our territory, the longer we will be condemned to watch from the sidelines and look at their constant enhancement."

A strategic surprise

In the months leading up to October 7, the former head of the Planning Division, Major General (Res.)
</item 12>

<item 13>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Iron Dome was so successful it not only supported the decision-making process but fundamentally changed it. Up until October 7, it served as a very successful
plaster to threats to a sovereign state, because what legitimacy would a state leader have for going to war when no citizens were harmed?"

Brigadier General Eran Ortal, until recently the commander of The Dado Center for Interdisciplinary Military Studies, a military research center subordinate to
the Operations Directorate, also believes that various technological measures such as the Iron Dome, and the barrier, influenced decision-making in Israel. In
his estimation, these helped the political echelon conduct a policy of military restraint towards the Gaza Strip, which in turn allowed Hamas to strengthen and
gain offensive capabilities.
</item 13>

<item 14>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"The Israelis must be totally losing it right now."

the strategic planning that must have gone into this against one of the most high tech occupation and surveillance systems in the world will be studied by many,
the Israelis must be totally losing it right now https://t.co/oUX2rokEjh — Rania Khalek (@RaniaKhalek) October 7, 2023

Tarik Cyril Amar said: "And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed."

And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed. — Tarik Cyril Amar (@TarikCyrilAmar) October 7,
2023

Mjoachim, another user, said that Israel's defence forces have lost their fame. He said the IDF lost an army base, a general was captured right out of his
house.
</item 14>

<item 15>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 15>

<item 16>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 16>

<item 17>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
A comprehensive study conducted on intelligence failures in the national arena, from the fall of the Iron Curtain in 1989 to 2010, showed no improvement in
providing or thwarting warnings about strategic surprises, even if the means of gathering information had improved. Four years after signing the study, former
US President Barack Obama admitted that US intelligence had not correctly assessed the severity of the threat of large parts of Iraq and Syria being occupied by
ISIS.

"No matter how technologically advanced your army or how effective your intelligence gathering system, identifying strategic surprises is a human process that
depends on analyzing what is called, in the intelligence community, 'weak signals,'" Barnea says. "These are pieces of relevant information that have been
conveyed to the organization, but because they are low-key, or because of the information overload around them, the organization fails to identify and interpret
them correctly."
</item 17>

<item 18>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Ortal points out that between Operation Pillar of Defense in 2012 and Operation Protective Edge in 2014, Israel significantly reduced targeted countermeasures
against terrorist operatives, and none focused on Hamas operatives. According to his analysis, this was because, prior to the Iron Dome coming into use in 2011,
the most dangerous threat posed by Gaza developed in the period preceding Operation Protective Edge. The understandings that Israel reached with Hamas in 2012
were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible. This, after air raid sirens sounded in Tel Aviv and
Jerusalem for the first time since the Yom Kippur War, and the education system shut down for more than a week.

Even after Iron Dome, Israel found it difficult to stand up against the significant disruption of daily life on the home front.
</item 18>

<item 19>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas' surprise attack on Israel has shocked many who are now questioning the country's famed intelligence agency Mossad, which is known for its
intelligence-gathering capability. For some, the belief that Mossad was among the best in the spying world was shattered today when Hamas carried out the
biggest coordinated attack in Israel and captured some territory in southern Israel. It is not the attack that has surprised people but the scale at which Hamas
operated and got control of some military bases and soldiers.

Also Read: Israel at war: Hamas cells, military compounds destroyed, US backs Tel Aviv; all you need to know

Younis Tirawi, who reports on Palestinian Affairs, said that Hamas militants had taken over Sderot police HQ and armored vehicles in the town. An analyst wrote
that some of the images coming out of Sderot over the last few hours were reaching degrees of horror "that I don't really think I can use words to properly
describe.
</item 19>

<item 20>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Giora Eiland, held a series of lectures for senior officers on the reasons for the strategic surprise that fell on the IDF in the Yom Kippur War. "Not one of
the officers imagined that 50 years after that mistake, it would repeat itself in exactly the same way," he tells Globes.

The strategic surprise that hit the State of Israel about four weeks ago bears similar characteristics to the Japanese attack on the US naval base at Pearl
Harbor in 1941; to the Argentine surprise attack on Britain's Falkland Islands in 1982; the Al-Qaeda attack on the United States on September 11, 2001; and to
the spread of ISIS throughout the Middle East in 2014.

In fact, Barnea claims, despite the technological progress and sophistication, intelligence officials are questioning whether there really is a trend towards
improvement of early detection and thwarting of strategic threats.
</item 20>
=== END ITEM id=bd4a038571d1be63 ===

=== ITEM id=ff9963fb4db5293a ===
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
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 1>

<item 2>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Brown noted that a 0.5% rise in Citrin Cooperman’s discount rate (i.e. raising it from 8.5% to 9.0%) would “reduce the fair value of the [HSF] portfolio by
7.9%, which on a leveraged basis is [approximately a] 10% reduction in NAV [Net Asset Value]”.

Should HSF’s board go down this route, he noted, he expected “the [HSF] shares to weaken… given that this represents another ‘unforced error’”.

“This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer,
Citrin Cooperman.”
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.
</item 3>

<item 4>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 4>

<item 5>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
There are a few clues in HSF’s latest public statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being
“majority-owned by funds managed and/or advised by Blackstone“.

In addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount rate’ used by
Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).

Citrin Cooperman continues to stick to an 8.5% discount rate in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to
move this 8.5% figure upwards to, say, 9% or 9.5%.

If Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.
</item 5>

<item 6>
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
</item 6>

<item 7>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
The reason for the delay? According to UK-listed HSF, it’s all to do with a recently-updated company valuation from HSF’s official independent valuer,
US-headquartered Citrin Cooperman (CC).

The HSF board says this valuation from CC was “materially higher than the valuation implied by proposed and recent transactions in the sector”.

HSF’s board particularly points to two transactions/would-be transactions as evidence:

The proposed sale earlier this year of assets to Blackstone-backed Hipgnosis Songs Capital for USD $440 million (pre-costs) – or $417.5 million (post-costs) – a
proposal that was ultimately rejected by HSF shareholders; The sale of 20,000 “non-core” songs from HSF’s portfolio to an unnamed buyer (possibly Kobalt Music
Group) for $23.1 million, as announced on December 11.
</item 7>

<item 8>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 8>

<item 9>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
It’s important to consider the past track record and management team before making investment decisions. Consultation with investment advisors who are
knowledgeable in this area can undoubtedly be beneficial.

The author is President & Head, Nuvama Wealth
</item 9>

<item 10>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Portfolio podcast: The art of valuing stocks Portfolio podcast: The art of valuing stocks

Let there be no doubt, valuation experts like Aswath Damodaran and many successful hedge fund managers believe that absolute valuation (which we discussed in
the Big Story in bl.portfolio edition dated October 8) is the better way to arrive at the value of a stock.

However, due to a combination of complexities involved in building appropriate models, insufficient information, time constraints and also the fact that there
are hundreds of stocks you want to consider, it is just not possible to do a DCF ( discounted cash flow model) for all your stock investments, and hence
relative valuation as an approach to picking stocks too is well accepted and widely prevalent.

When you decide to buy or sell a stock based on its PE or P/B or EV/EBITDA or EV/Revenue or even EV/subscriber ratios, you are engaging in the art of relative
valuation.
</item 10>

<item 11>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.
</item 11>

<item 12>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 12>

<item 13>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 13>

<item 14>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 14>

<item 15>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Christopher Brown, JP Morgan, on the HSF board statement today (December 19)

In addition, Brown expressed his surprise that HSF asked HSM to “opine on the valuation given [HSM’s] conflict of interest”.

His point: If HSM turned around and said, “yep, HSF’s valuation is 25% too high, better bring it down”, then it would produce a new valuation for HSF that may
allow Blackstone (or the Blackstone-controlled Hipgnosis Song Capital) to launch an acquisition bid for the HSF portfolio. (Or, at least, a part of that
portfolio.)

The likely price for this hypothetical Blackstone bid would be smaller than the failed $440 million ($418.5 million post-costs) bid that HSC (via HSM) made for
a chunk of the HSF portfolio earlier this year.
</item 15>

<item 16>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 16>

<item 17>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
HSF’s investments in seasonal Christmas ‘evergreens’ continue to pay off too, with cuts on Michael Buble’s Christmas album and – most famously – a share of
Mariah Carey’s perennial megahit, All I Want For Christmas Is You.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 17>

<item 18>
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
</item 18>

<item 19>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Not to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for overseeing the valuation of the
Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”

WHAT’S really going on?

It’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director – is making something
of a ‘hidden’ statement to shareholders with today’s announcement.

It may be a coded attempt by HSF’s board to gnash teeth towards HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the
exclusive power to sweep in and acquire HSF’s portfolio under several different circumstances.
</item 19>

<item 20>
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
</item 20>
=== END ITEM id=ff9963fb4db5293a ===

=== ITEM id=6718f7f86609d962 ===
Gold sentences and facts:
Sentence 0: Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to keep their hands off of some social media posts.
  Fact 0.0: The Supreme Court decided last week that it would hear the pair of cases.
  Fact 0.1: The pair of cases revolve around Republicans crafting state-specific laws.
  Fact 0.2: The state-specific laws order platforms to keep their hands off of some social media posts.
Sentence 1: Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation.
  Fact 1.0: Under Musk, Twitter has prioritized free speech.
  Fact 1.1: Twitter is now called X.
  Fact 1.2: Under Musk, free speech on Twitter is managed by crowdsourced-based moderation.

Retrieved items:
<item 1>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 1>

<item 2>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 2>

<item 3>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
During this period the DSA had not entered into force yet for VLOPs and VLOSEs, including X/Twitter. The Commission cannot comment further on the substance of
the risk assessments. We are analysing the available information on a range of issues and we are in close contact with X but we cannot comment on specifics and
timelines. As stated by Commissioner [Thierry] Breton, minor protection and integrity of elections are priorities in the enforcement of the DSA.

Penalties for breaches of the DSA can scale up to 6% of global annual turnover and the Commission has previously wanted that not following the Code of Practice
on Disinformation would be factored into its assessments. So the price for X ignoring the EU’s warnings on disinformation could be steep — albeit its ad revenue
has cratered since Musk’s takeover, as a result of his failure to tackle toxicity and harmful disinformation encouraging an advertiser exodus — and shrinking
earnings would effectively downsize any future penalties X could face under the DSA.
</item 3>

<item 4>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 4>

<item 5>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 5>

<item 6>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
It’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted digital rulebook, the Digital
Services Act (DSA).

Announcing the opening of a “formal proceeding” today, the European Commission said the investigation will look at whether the social networking platform may
have breached rules linked to risk management, content moderation, dark patterns, advertising transparency and data access for researchers.

The opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights group, noyb — although today’s formal
proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of requests for information. Its earlier
actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.

Back in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising out of
the conflict.
</item 6>

<item 7>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 7>

<item 8>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Subscription revenues to a revamped Twitter Blue paid offering (now called X Premium) don’t appear to have been substantial enough to plug the ad gap — which
may explain Musk’s recent suggestion he plans to charge all users a small fee. However ending free access to X could be the final nail in the coffin of the
platform as a real-time information network if it triggers a further/final exodus of remaining legacy users whose knowledge and engagement lent Twitter its
original value.

Disinformation and spammers may rush in to fill the gap, generating a parody of activity, but without a critical mass of genuine users to pay attention and
contribute value and vitality — aka the life-blood of a bona fide information network — X really will mark the spot of Twitter’s demise.

Also today, a report by misinformation watcher NewsGuard found that a decision by Musk to remove labels denoting state-run or government-affiliated media
accounts led to a surge in propaganda spreading on X, with Russia’s RT seeing the biggest gains per the analysis.

From the report:
</item 8>

<item 9>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom recommends these platforms implement policies like not allowing strangers to send direct messages to children, using hash matching to detect and remove
CSAM, maintaining content and search moderation teams, and offering ways for users to report harmful content.

Large tech platforms already follow many of these practices, but Ofcom hopes to see them implemented more consistently. “We think they represent best practice
of what’s out there, but it’s not necessarily applied across the board,” Whitehead says. “Some firms are applying it sporadically but not necessarily
systematically, and so we think there is a great benefit for a more wholesale, widespread adoption.”

There’s also one big outlier: the platform known as X (formerly Twitter). The UK’s efforts with the legislation long predate Elon Musk’s acquisition of Twitter,
but it was passed as he fired large swaths of its trust and safety teams and presided over a loosening of moderation standards, which could put X at odds with
regulators.
</item 9>

<item 10>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 10>

<item 11>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
EU turns up the heat on X over illegal content in wake of Israel-Hamas war

The European Union is cranking up the heat on Elon Musk-owned X. Late Thursday, the Commission sent the company a formal request for more information — after
issuing a public warning Wednesday about reports that illegal content and disinformation targeting the Israel-Hamas war is circulating on the platform, and
Twitter’s CEO sending a high-level (non specific) response to that warning.

The move could prefigure the opening of a formal investigation of X under the bloc’s content moderation rulebook, the Digital Services Act (DSA). If the
Commission moves ahead on that, it will be the first investigation opened under the DSA since a compliance deadline for so called “very large online platforms”
(aka VLOPs) kicked in this summer.

Non-compliance under DSA rules, meanwhile, could trigger fines of up to 6% of annual turnover, plus potential blocking of a service for repeated infringements.
X owner Elon Musk last projected revenues of about $3 billion for this year.
</item 11>

<item 12>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Reached for a response to the EU probe, X’s Joe Benarroch — who was brought into the company by CEO Linda Yaccarino relatively recently to work on business
operations — said:

X remains committed to complying with the Digital Services Act, and is cooperating with the regulatory process. It is important that this process remains free
of political influence and follows the law. X is focused on creating a safe and inclusive environment for all users on our platform, while protecting freedom of
expression, and we will continue to work tirelessly towards this goal.

While X under Musk has very evidently been flying the ship in the opposite direction to the responsible governance the DSA intends to encourage — hence
attracting so many warnings from EU lawmakers, even long before today’s formal probe — it’s notable X recently started offering a research program for outsiders
to get data to systemic risks arising on the platform in the EU. However the Commission obviously has doubts X has gone far enough on the transparency front to
meet the DSA’s bar.
</item 12>

<item 13>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 13>

<item 14>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 14>

<item 15>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 15>

<item 16>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU said the probe will also look at the effectiveness of “related policies mitigating risks to civic discourse and electoral processes”

— with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated views of users, who may vote to label disputed
tweets with additional context. The EU said the probe will also look at the Transparency requirements — with a focus on “suspected shortcomings” by X in giving
researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings in its ads repository (aka the ads transparency
library the regulation also mandates)

— with a focus on “suspected shortcomings” by X in giving researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected
shortcomings in its ads repository (aka the ads transparency library the regulation also mandates) Suspected deceptive design elements in X’s user interface —
“notably in relation to checkmarks linked to certain subscription products, the so-called Blue checks”

“If proven,
</item 16>

<item 17>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation. Its policies have resulted in increased hate speech
and toxic content on the platform, some studies have shown.

Mozilla believes there’s a better way forward and, more importantly, that now is exactly the right time to invest in that path. And, as a wholly owned
subsidiary of a nonprofit, the company says it’s not motivated by generating earnings for shareholders or returning a VC investment, allowing it to progress
with a collaborative approach where it takes in input from a lot of different voices.

In an interview with TechCrunch, Mozilla Senior Director of Content Carolyn O’Hara explained why Mozilla has taken an interest in the fediverse and Mastodon,
specifically, and how it expects to experiment in the months ahead.
</item 17>

<item 18>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
To explain the Supreme Court’s decision to wade into these issues — and what happens next — TechCrunch spoke to Paul Barrett, NYU adjunct law professor and
Deputy Director of NYU Stern’s Center for Business and Human Rights.

Why is the Supreme Court involved?

These cases actually started in Florida and Texas a few years ago before wending their way to the Supreme Court this year. In those two states, Republican
lawmakers passed parallel laws to control how social media companies operate. In Florida, Governor Ron DeSantis signed Senate Bill 7072 into law in May 2021. In
Texas, House Bill 20 made it through the state legislature and was signed by Governor Greg Abbott in September 2021.

“The reason why these cases are in front of the Supreme Court is actually relatively simple: Florida and Texas were more or less the first out of the gate in
imposing this type of restriction on social media companies,” Barrett explained.
</item 18>

<item 19>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
They must also clearly communicate their T&Cs to users and be able to demonstrate proper enforcement.

Larger platforms like X also have additional obligations to identify and mitigate systemic risks such as disinformation, gender-based violence or negative
effects on the exercise of fundamental rights.

Additionally, the regulation includes a “crisis response” mechanism (Article 36) which enables the Commission to adopt rapid-fire measures on larger platforms
in situations of “serious threat”, such as war.

Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show
Hamas missile attacks on Israel but was actually footage from a video game.

Meanwhile X’s ability to respond internally to reports of content problems have been drastically pared back after Musk’s takeover last year — which saw major
layoffs, including in content moderation and human rights as part of the new billionaire owner’s bid to improve the platform’s profitability.
</item 19>

<item 20>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“In this context, 832 videos related to the war have been fact-checked, of which 211 have been removed,” Jourova noted.

The EU also flagged reporting by Microsoft that told it Bing Search had either promoted information or downgraded questionable information in relation to almost
800,000 search queries related to the Ukraine crisis.

Jourova’s speech also highlighted a couple of other areas where she urged Code signatories to go further — calling (yet again) for more consistent moderation
and investment in fact-checking, especially in smaller Member States and languages.

She also criticized platforms over access to data, saying they must step up efforts to make sure researchers are empowered to scrutinize disinformation flows
“and contribute to the necessary transparency”.

Both are areas where X/Twitter under new owner, Elon Musk, has moved out of step with EU expectations on countering disinformation.

Twitter (now X) was an original signatory to the disinformation Code but Musk took the platform out of the initiative back in May, as critical scrutiny of his
actions dialled up in the EU.
</item 20>

<item 21>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 21>

<item 22>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 22>

<item 23>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What’s at stake in the Supreme Court’s landmark social media case The court will wade into First Amendment questions with major implications for platforms

The Supreme Court is poised to decide whether a pair of state laws are allowed to reshape the ability of social media companies to control what does — and
doesn’t — appear on their platforms.

Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to
keep their hands off of some social media posts. Since the early days of the Trump administration, Republicans have accused social media companies of
deliberately suppressing conservative viewpoints.

While research has not supported these claims, researchers have demonstrated that conservative social media users are disproportionately exposed to political
misinformation, a phenomenon that could explain anecdotal claims of ideologically lopsided enforcement on social platforms.

Whether animated by those perceived differences or the political perks of accusing social platforms of anti-conservative bias, conservative lawmakers in Florida
and Texas passed laws to restrict how those companies are allowed to moderate content.
</item 23>

<item 24>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 24>

<item 25>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
It’s the only way to fight bots without blocking real users. This won’t stop bots completely, but it will be 1000X harder to manipulate the platform. — Elon
Musk (@elonmusk) October 18, 2023

The plan to charge X new users $1 per year, currently only live in test markets of New Zealand and the Philippines, is the latest significant change to the
network formerly known as Twitter since its acquisition by Elon Musk, who has revamped its subscription offering, refocused efforts on supporting creators and
rebranded the network as “X,” as part of his broader plan to turn the social network into an everything app encompassing text, live and recorded, audio and
video calling, payments and more.

However, the timing of the bot-countering measure comes at a time when X is facing more competition than ever before. As Musk’s chaotic product and policy
changes have taken root, some X users abandoned the platform for rivals or decreased their posting.
</item 25>

<item 26>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What does this have to do with the First Amendment?

The case revolves around First Amendment rights — but, counterintuitively, it’s the rights of social media companies that are in question, not the rights of
their users.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment?”

“These cases are about the First Amendment and how the First Amendment applies to social media companies. And then more specifically, what the First Amendment
has to say about content moderation, which is obviously a subset, although a really big and important subset of what social media companies do,” Barrett said.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment — or what you also
might call content moderation — in sorting out what expression does and does not appear on the platforms that they own? So it’s just not clear right now what
precisely the First Amendment says about that question.”

Are the Texas and Florida laws identical?
</item 26>

<item 27>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 27>

<item 28>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 28>

<item 29>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 29>

<item 30>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 30>
=== END ITEM id=6718f7f86609d962 ===
