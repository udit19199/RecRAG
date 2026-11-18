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

=== ITEM id=b65d7296336b17c1 ===
Gold sentences and facts:
Sentence 0: Back then, just like today, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 0.0: Back then, home prices had boomed for years.
  Fact 0.1: Today, home prices have boomed, just like back then.
  Fact 0.2: Fed officials were ultimately forced to hike interest rates aggressively.
  Fact 0.3: The Fed hiked interest rates in an attempt to fight inflation.
Sentence 1: Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
  Fact 1.0: Postponements of such reports could complicate things for the Fed.
  Fact 1.1: The Fed has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.

Retrieved items:
<item 1>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, financial system, Fed, The Sydney Morning Herald, US, economy, interest rates, ASX to rise after interest rate hopes boost US
shares, Expedia Group, Treasury yields, Katie Nixon, Northern Trust Wealth Management, investments, coronavirus crash, three high-profile US bank failures
</item 1>

<item 2>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> jobs
- Want to help your kids achieve their potential? This expert has some advice -MENTIONS-> jobs
- Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' -MENTIONS-> jobs
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> jobs
- Suleyman -SAID-> jobs

Entities: jobs, ASX set to open higher as Wall Street rebounds; $A rises, wages, ADP, ADP report, comprehensive jobs report, markets, overnight rate, private
employers, US government, Federal Reserve, The Age, interest rates, oil prices, ASX
</item 2>

<item 3>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Reserve Bank
- Jessica Amir -SAID-> Reserve Bank
- ASX set to drop as Wall Street’s September slump deepens -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set for bright start despite Wall Street slip; $A drops -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set to open lower as Wall Street closes September with more losses -PUBLISHED_BY-> The Sydney Morning Herald

Entities: Reserve Bank, The Sydney Morning Herald, Australia, ASX, Wall Street, ASX set to rise ahead of inflation report after big tech lifts Wall Street,
Infratil, Meridian Energy, Resmed, GSFM, Stephen Miller, Reserve Bank board meeting, Interest rate rise, Equity market, REITS
</item 3>

<item 4>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set to open lower as Wall Street closes September with more losses -PUBLISHED_BY-> The Sydney Morning Herald
- The best shows to stream in December -PUBLISHED_BY-> The Sydney Morning Herald
- McRae reveals hidden message he carried under collar of his grand final shirt -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set to drop as Wall Street’s September slump deepens -PUBLISHED_BY-> The Sydney Morning Herald

Entities: The Sydney Morning Herald, ASX to rise after interest rate hopes boost US shares, US shares, Clark, Westpac, NAB, ANZ, interest-rate trade, long-term
bond yields, employment data, growth stocks, rate rise, S&P 500, Wall Street, ASX
</item 4>

<item 5>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Evolution
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Brent crude
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> benchmark US crude
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Brent crude
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> gold miners

Entities: benchmark US crude, Brent crude, ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields, Thursday’s inflation report,
Seema Shah, Principal Asset Management, Fed’s next move on interest rates, interest rate hike, Fed, The Age, the economy, bond yields, 10-year Treasury yield,
crude oil prices, ASX
</item 5>

<item 6>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 6>

<item 7>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, Apple, Bank of America, Fed, The Sydney Morning Herald, US, ASX to rise after interest rate hopes boost US shares, Wall Street,
BofA Global Research, Institute for Supply Management, US services industries
</item 7>

<item 8>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> stock market
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> stock market
- Noff -SAID-> stock market
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets

Entities: stock market, financial markets, ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields, 30-year Treasury bonds, auction
of 30-year Treasury bonds, Federal Reserve, interest rate, The Age, economy, bond market, ASX, Wall Street, inflation report, inflation
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

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, The Sydney Morning Herald, US, economy, bond market, interest rates, ASX to rise after interest rate hopes boost US shares, ASX, US
shares, Wall Street, Treasury yields, The Fed, job market, pay raises, Friday’s jobs report
</item 9>

<item 10>
Graph query row:
article: ASX set for bright start despite Wall Street slip; $A drops
published_relationship: PUBLISHED_BY
publisher: The Sydney Morning Herald
Federal_Reserve_relationship: MENTIONS
Federal_Reserve: Federal Reserve
</item 10>
=== END ITEM id=b65d7296336b17c1 ===

=== ITEM id=bc4bb12f69ef33cf ===
Gold sentences and facts:
Sentence 0: In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.
  Fact 0.0: The Wallabies had a World Cup flop.
  Fact 0.1: The Queensland Rugby Union reaffirmed its position.
  Fact 0.2: The Queensland Rugby Union is committed to moving to a centralised high-performance model.
  Fact 0.3: The Queensland Rugby Union will not follow the NSW Waratahs by handing over the keys to Rugby Australia.
  Fact 0.4: The NSW Waratahs handed over the keys to Rugby Australia.
Sentence 1: He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.
  Fact 1.0: The person will also be tasked with 'integrating' the programs.
  Fact 1.1: Rugby Australia is trying to centralise all of the Super Rugby high-performance programs.

Retrieved items:
<item 1>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, RA, QRU, ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop, RWC, Nine
Entertainment, Stan Sport, The Roar
</item 1>

<item 2>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
“The very strength of our State unions is the expertise they bring in their own markets.

Advertisement

“We are open to work on commercial initiatives together but will always retain our independence for the benefit of Queensland Rugby, its members and
stakeholders.

“Queensland Rugby has built a sustainable business model through the elimination of legacy operating debt, delivered four-straight operating profits and worked
in partnership with Governments to successfully deliver $31 million of improvements to Ballymore through the construction of the National Rugby Training Centre,
the home of the Wallaroos and Queensland Reds.”

The letter also pointed to the 71,000 players across the state, including more than 16,000 of those female participants, 257 schools featuring rugby as part of
its program, as well as its Hospital Cup competition and regional club competitions, as evidence that the game was in good health in Queensland.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, rugby, Queensland, governments, RA, ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’
RWC flop, Queensland Rugby, State unions, Ballymore, National Rugby Training Centre, Wallaroos, Queensland Reds, Hospital Cup competition, regional club
competitions, RWC
</item 2>

<item 3>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised
high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.

They also expressed their strong support for Rugby Australia to conduct a “fully independent and transparent top-to-bottom review of the Wallabies program and
how it can be improved” in light of the historic first World Cup pool exit.

They added: “It should also provide a clear view of what is needed in a leadership and governance framework to ensure we put the right people, in the right
roles, in the right system.”

Facebook Twitter Whatsapp Reddit Email Share

The QRU has also expressed their desire for Rugby Australia to reinstate the annual $1.7 million in funding to each of the Super Rugby franchises that was
withdrawn following the Covid pandemic to help keep the game afloat.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, Covid pandemic, Wallabies, Queensland Rugby Union, NSW Waratahs, Rugby Australia, ‘Right people in right roles’: QRU breaks silence on RA’s
reform hopes, backs review into Wallabies’ RWC flop, Super Rugby franchises, The Roar
</item 3>

<item 4>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The Melbourne Rebels are expected to follow.

RA chief executive Phil Waugh says building trust back between the governing body and states is essential to their plans.

Under fire Wallabies coach Eddie Jones also advocated for Australian rugby to move to a centralised model, believing the nation had fallen behind the rest of
the world.

“We’re a state-based union that’s run by the states, the national union wants to assume more responsibility and when things were going well, no one questioned
that,” Jones said.

Advertisement

“But over the last period of time, they haven’t been going so well and this tournament’s probably been the icing on the cake where it’s got to a stage where I
think people, most people probably think there’s time for change. But it’s not going to go easy.

“[It’s] one of the key issues. We need to maximize the development of the players.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, Eddie Jones, RA, Phil Waugh, Melbourne Rebels, centralised model, building trust back between the governing body and states, development of
the players, QRU, ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop, RWC, Australian rugby, The Roar
</item 4>

<item 5>
Graph query row:
article: ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop
queensland_rugby_union: Queensland Rugby Union
rugby_australia: Wallabies
passage: ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised
high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.

They also expressed their strong support for Rugby Australia to conduct a “fully independent and transparent top-to-bottom review of the Wallabies program and
how it can be improved” in light of the historic first World Cup pool exit.

They added: “It should also provide a clear view of what is needed in a leadership and governance framework to ensure we put the right people, in the right
roles, in the right system.”

Facebook Twitter Whatsapp Reddit Email Share

The QRU has also expressed their desire for Rugby Australia to reinstate the annual $1.7 million in funding to each of the Super Rugby franchises that was
withdrawn following the Covid pandemic to help keep the game afloat.
</item 5>

<item 6>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the open letter to their members on Wednesday morning, QRU chief executive David Hanham and chairman Brett Clark moved to update constituents on the reform
measures that RA is attempting to usher through the game.

Last week, The Roar revealed that the New South Wales Rugby Union was in the latter stages of handing over their license to RA.

By doing so, the NSW Waratahs, who were on the brink of insolvency, handed over their commercial and high-performance property.

Advertisement

The decision will see all male and female Waratahs players, including coaches and high-performance staff, as well as all commercial and operations staff, run
out of RA’s office.

NSW Waratahs chief executive Paul Doorn informed stakeholders of the decision last week.

RA is cognisant that not every Super Rugby franchise will follow suit immediately, but the governing body is hopeful that now one of the two biggest states has
agreed to hand over the keys, others will follow suit swiftly.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, RA, David Hanham, Brett Clark, New South Wales Rugby Union, NSW Waratahs, Paul Doorn, review into Wallabies’ RWC flop, Super Rugby, QRU,
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop, RWC, The Roar
</item 6>

<item 7>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Horne briefly worked at Saracens as a general manger.

RA always said they wanted to have the role filled by Christmas, with chief-executive Phil Waugh saying the successful candidate had one of the biggest jobs to
change the game’s standing in Australia.

“It’s a big job,” Waugh said. “It’s the biggest job on my leadership team in terms of the impact it has on the general sentiment around the game.”

Horne will oversee all of Australia’s men’s and women’s high-performance programs.

Advertisement

He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.

Perhaps most pressing will be building a list of coaching candidates to take over from Eddie Jones, with RA wanting to have the new Wallabies coach set in stone
by the start of Super Rugby.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, Australia, Eddie Jones, RA, Phil Waugh, Horne, Super Rugby, Saracens, high-performance programs, RA set to announce local as
high-performance director – and how it could help land Schmidt as Wallabies coach, Schmidt, The Roar
</item 7>

<item 8>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, Australia, AFL, RA, rugby league, Australian Super Rugby sides, QRU, ‘Right people in right roles’: QRU breaks silence on RA’s reform
hopes, backs review into Wallabies’ RWC flop, ACT Brumbies, The Roar
</item 8>

<item 9>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU also expressed its satisfaction that RA was willing to discuss the annual $1.7m being reinstated, which had been withheld because of the Covid pandemic
to keep the game afloat.

Meanwhile, The Roar understands the ACT Brumbies have sent RA confirmation of solvency for the next 12 months.

RA was forced to take out a $40m loan from Ares to keep the game afloat.

Advertisement

While RA has around another $15m left to spend from the loan, the governing body is looking to borrow up to another $60m so they’re in the strongest possible
position to capitalise on the two major money-making events – the 2025 British and Irish Lions series and home men’s World Cup in 2017 – and succeed.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, RA, QRU, ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop, ACT Brumbies, Ares,
Covid pandemic, 2025 British and Irish Lions series, home men’s World Cup in 2017, The Roar
</item 9>

<item 10>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU confirmed that on Wednesday morning.

Advertisement

“The QRU is committed to working with Rugby Australia to lead reform of our game at a high-performance level, and help Rugby Australia develop an improved game
through high-performance centralisation,” the statement read.

“Centralisation” would include a national player-contracting model, and alignment of key high-performance staff, such as coaching, strength and conditioning,
and medical to operate as one national system.”

The letter pointed to RA’s involvement in Les Kiss’ appointment as Reds coach, with the former State of Origin winger turned coach, who has spent the best part
of two decades coaching in Ireland and England, replacing Brad Thorn.

As The Roar has previously reported, the QRU stopped short of wanting to hand over its commercial property.

“While the QRU is very supportive of high-performance alignment, it will not agree to any proposal on centralising commercial or corporate functions,” the
letter read.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Ireland
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Ireland
- The best VPN services for 2023 -MENTIONS-> Ireland
- WebSummit derailed by founder’s public fight with those supporting Israel in Hamas conflict -MENTIONS-> Ireland
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ireland

Entities: Ireland, Reds, England, Wallabies, QRU, Rugby Australia, Les Kiss, Brad Thorn, Les Kiss’ appointment as Reds coach, reform, high-performance
centralisation, commercial or corporate functions, review into Wallabies’ RWC flop, ‘Right people in right roles’: QRU breaks silence on RA’s reform hopes,
backs review into Wallabies’ RWC flop, The Roar
</item 10>
=== END ITEM id=bc4bb12f69ef33cf ===

=== ITEM id=1862de397a2f9967 ===
Gold sentences and facts:
Sentence 0: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 0.0: This brings us to the fated sighting of Swift at Arrowhead Stadium on Monday.
  Fact 0.1: Swift was cheering on a player numbered 'No.' (the sentence is cut off).
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium herself.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has a new relationship with Kelce.
  Fact 2.1: Swift has shared that she has nothing to hide.
  Fact 2.2: Swift has said she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Taylor Swift, Travis Kelce, NFL, first game, Taylor Swift reveals secret start to her relationship with Travis Kelce, The
Independent, Kansas City Chiefs
</item 1>

<item 2>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Fanatics, Chiefs, Reynolds, jersey sales, The Independent, Travis Kelce admits the NFL is overdoing it with their
Taylor Swift coverage, People, Jackman, Deadpool, Wolverine, NFL statement, bios and profile imagery
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Arrowhead Stadium, Thursday Night Football, Jason, Taylor Swift, rumored romance, Travis Kelce, Al Michaels, Chiefs-Broncos, White Horse,
celebrities, Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, Broncos
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> All Too Well
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> All Too Well
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Argentina
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Argentina
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Argentina

Entities: All Too Well, Argentina, Buenos Aires, Karma, Time, Taylor Swift, Travis Kelce, Chiefs, Taylor Swift reveals secret start to her relationship with
Travis Kelce, Eras Tour show, Style, The Independent
</item 4>

<item 5>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Taylor Swift, Travis Kelce, Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift
romance, Kansas City Chiefs game against the Los Angeles Chargers, Chariah Gordon, Mecole Hardman Jr, Instagram, Los Angeles Chargers, The Independent, Kansas
City Chiefs
</item 5>

<item 6>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Getty Images, New York Jets, Chiefs, Taylor Swift reveals secret start to her relationship with Travis Kelce, game
between the Chiefs and New York Jets, Denver Post, Super Bowl LVII, The Independent, football
</item 6>

<item 7>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

In addition to the kiss, the football star also had the perfect reaction to a fan calling him Swift’s boyfriend. After a fan shouted: “Let’s go Taylor’s
boyfriend” while Kelce was waiting for an elevator, he responded with a small smile on his face and by pumping his fist into the air.

Aside from their appearances at sports events, Swift and Kelce have also been photographed together on other occasions, including a date earlier this month in
New York City after their surprise appearances on the season 49 premiere of Saturday Night Live. At the time, sources alleged to Entertainment Tonight that,
during the date, Kelce got a little too close to Swift’s security personnel and allegedly told her bodyguard to step aside.

However, during an episode of his podcast, Kelce explained what he said to the security, and clarified that he didn’t push the bodyguard.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> New York City
- The people who ruined the internet -MENTIONS-> New York City
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> New York City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> New York City
- The best video games of the year so far -MENTIONS-> New York City

Entities: New York City, Taylor Swift, Travis Kelce, Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance,
Saturday Night Live, season 49 premiere of Saturday Night Live, Entertainment Tonight, The Independent
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."

Graph facts:
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Taylor Swift -MENTIONS-> jersey sales

Entities: Taylor Swift, Arrowhead, rumored romance, Travis Kelce, Mahomes, postgame press conference, Swiftie talk, Taylor Swift-Travis Kelce timeline:
Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, game, Donna Kelce
</item 8>

<item 9>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Jason Kelce
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Jason Kelce
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Jason Kelce
- Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jason Kelce
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Jason Kelce

Entities: Jason Kelce, Taylor Swift, Travis Kelce, New York Jets, The Independent, Kansas City Chiefs, National Football League, game against the New York Jets,
New Heights with Jason and Travis Kelce, Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage
</item 9>

<item 10>
Graph query row:
article: A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation
publisher: The Independent
passage: A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
connected_entity: None
relationship: None
</item 10>
=== END ITEM id=1862de397a2f9967 ===
