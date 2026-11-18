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

=== ITEM id=2d917aa729a71fab ===
Gold sentences and facts:
Sentence 0: However, despite Floyd’s massive net worth, Canelo has inched closer to Floyd’s billionaire status over the years.
  Fact 0.0: Floyd has a massive net worth.
  Fact 0.1: Canelo has inched closer to Floyd's billionaire status over the years.
Sentence 1: He was the richest person in the world under 30 [and] he was going to spend it to save humanity from extinction,” he said.
  Fact 1.0: The speaker said he was the richest person in the world under 30.
  Fact 1.1: The speaker said he was going to spend his wealth to save humanity from extinction.

Retrieved items:
<item 1>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
It would also be a money magnet that would funnel income from a variety of channels.

A few of those lucrative channels of income would be revenue from pay-per-view sales, live gate revenue, merchandise sales from either party and various
promotional events from the fight. What’s more, if anyone wondered if people would want to watch the fight. The attention it has gathered in the small span of
their X (Twitter) feud should be evidence enough. Not to mention, both fighters command a massive fan base, counted in millions across the globe.

Canelo Alvarez and McGregor’s history of massive purses

If Canelo’s PPV pull was ever in question, his history of fight purses should convince everyone. Stepping back in time, Canelo’s fight with Mayweather collected
2.2 million pay-per-view buys. His 2017 bout against Julio César Chávez Jr. collected 1 million PPV buys.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> X
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> X
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> X
- Is Google Search better than the rest? And is that fair? -MENTIONS-> X
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> X

Entities: X, Floyd Mayweather, Julio César Chávez Jr., Canelo’s fight with Mayweather, Canelo’s 2017 bout against Julio César Chávez Jr., Potential Conor
McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status, Conor McGregor, Canelo Alvarez,
Essentially Sports
</item 1>

<item 2>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
collected 1 million PPV buys. Canelo’s first battle with Gennady Golovkin gathered 1.3 million PPV sales. In his most recent battle against Jermell Charlo, the
fight collected 700K PPV sales. These were just to name a few; over the years, Canelo has fought in increasingly profitable bouts.

On the other hand, Conor McGregor has created his own history in MMA as well as boxing. First of all, McGregor’s fight against Mayweather generated 4.3 million
pay-per-view buys. His second fight against Dustin Poirier collected 1.6 million buys. The infamous fight against Khabib Nurmagomedov saw 2.4 million PPV sales.
And lastly, his second fight against Nate Diaz collected 1.65 million PPV buys. If these numbers are any evidence, a Canelo vs.

Graph facts:
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -PUBLISHED_BY-> Essential Sports
- Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? -PUBLISHED_BY-> Essential Sports
- Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? -PUBLISHED_BY-> Essential Sports
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -PUBLISHED_BY-> Essential Sports
- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled Lennox Lewis -MENTIONS-> Floyd Mayweather

Entities: Essential Sports, Floyd Mayweather, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net
Worth and His Billionaire Status, Conor McGregor, Gennady Golovkin, Dustin Poirier, Khabib Nurmagomedov, Nate Diaz, Canelo’s first battle with Gennady Golovkin,
Canelo’s most recent battle against Jermell Charlo, Conor McGregor’s fight against Floyd Mayweather, Conor McGregor’s second fight against Dustin Poirier, Conor
McGregor’s fight against Khabib Nurmagomedov, Conor McGregor’s second fight against Nate Diaz, Jermell Charlo
</item 2>

<item 3>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
Furthermore, Canelo and McGregor’s careers are intertwined with the legendary Floyd Mayweather. ‘Money’ faced both fighters at one point or the other, leaving
each with a boxing lesson. When Mayweather defeated Canelo, it pushed him to hone his skills while starting a lifelong rivalry. On the other hand, McGregor made
his boxing debut against Mayweather, losing the fight but leaving with a fat stack.

Implications of a fight between Canelo Alvarez and Conor McGregor

The excitement and allure of two mammoths of combat sports getting in the ring to pummel each other to the brink of their endurance is undeniable. Not to
mention, the fight will leave either party significantly wealthier than before. A potential clash between Saul Canelo Alvarez and the ‘Notorious’ Conor McGregor
would not just be a dream come true for fans everywhere. It would also be a money magnet that would funnel income from a variety of channels.
</item 3>

<item 4>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
The Aces defeated the Liberty 98-81. NOTE TO USER: User expressly acknowledges and agrees that, by downloading and or using this photograph, User is consenting
to the terms and conditions of the Getty Images License Agreement. (Photo by Ethan Miller/Getty Images)

However, despite Floyd’s massive net worth, Canelo has inched closer to Floyd’s billionaire status over the years. If Canelo vs. McGregor ever manages to
materialize, it could help Canelo take a significant leap to Mayweather’s net worth. The financial windfall from such a humongous clash would be substantial,
driving him closer to Mayweather’s level of wealth. If Canelo’s past earnings are any evidence, this fight will be bigger than ever before. Canelo’s biggest
fight to date is $45 million from the third encounter with Golovkin.
</item 4>

<item 5>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
Later, the banter further escalated, bringing a potential fight in the picture.

Canelo Alvarez’s earnings and Floyd Mayweather’s influence

ADVERTISEMENT Article continues below this ad

Canelo Alvarez reached his career high by becoming the first boxer to become the undisputed champion at super-middleweight by holding onto the WBA Super, WBC,
and Ring Magazine super middleweight titles since 2020, and the IBF nd WBO titles since 2021. Meanwhile, his pockets became deeper and deeper, to the point that
Canelo boasts a $180 million net worth in 2023. The most significant addition to his bank balance came when he signed a lucrative $365 million deal with DAZN
and Golden Boy Promotions.

Graph facts:
- Floyd Mayweather -PARTICIPATED_IN-> Canelo’s fight with Mayweather
- Showtime Boxing to take final 10-count: Top 5 greatest fights revisited -MENTIONS-> Floyd Mayweather
- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled Lennox Lewis -MENTIONS-> Floyd Mayweather
- Floyd Mayweather -PARTICIPATED_IN-> Floyd Mayweather’s 2013 performance against Canelo Alvarez
- Floyd Mayweather -PARTICIPATED_IN-> Conor McGregor’s fight against Floyd Mayweather

Entities: Floyd Mayweather, DAZN, Golden Boy Promotions, $365 million deal, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to
Surpass Floyd Mayweather’s Net Worth and His Billionaire Status, Conor McGregor, Canelo Alvarez, WBC, Essentially Sports, IBF, Ring Magazine, WBA, WBO
</item 5>

<item 6>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
If these numbers are any evidence, a Canelo vs. McGregor clash would break all their previous records by a mile.

A substantial step toward Mayweather’s net worth/Billionaire status

It’s no secret that Floyd Mayweather is the richest boxer in the world. The legendary undefeated boxer boasts a staggering net worth of $450 million. It makes
the retired boxer the richest personality in boxing to ever exist. Despite retiring from the sport, he occasionally fights in exhibition matches, generating
millions each time. In addition, Floyd has made over a billion dollars in the span of his career.

ADVERTISEMENT Article continues below this ad

via Getty LAS VEGAS, NEVADA – JUNE 29: Floyd Mayweather Jr. attends a game between the New York Liberty and the Las Vegas Aces at Michelob ULTRA Arena on June
29, 2023 in Las Vegas, Nevada. The Aces defeated the Liberty 98-81.

Graph facts:
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -PUBLISHED_BY-> Essential Sports
- Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? -PUBLISHED_BY-> Essential Sports
- Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? -PUBLISHED_BY-> Essential Sports
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -PUBLISHED_BY-> Essential Sports
- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled Lennox Lewis -MENTIONS-> Floyd Mayweather

Entities: Essential Sports, Floyd Mayweather, net worth, billionaire status, Nevada, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer
to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status, Conor McGregor, Canelo vs. McGregor clash, New York Liberty vs. Las Vegas Aces game, New
York Liberty, Las Vegas Aces, Canelo Alvarez, boxing, Michelob Ultra Arena, Las Vegas
</item 6>

<item 7>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
via Imago RECORD DATE NOT STATED Saul -Canelo- Alvarez MEX vs Jermell Charlo USA – Weigh-in ceremony Saul -Canelo- Alvarez MEX during the weigh-in ceremony
prior to the Boxing fight against Jermell Charlo USA, at the T-Mobile Arena Explanade on September 29, 2023. brbr Saul -Canelo- Alvarez MEX durante la ceremonia
de pesaje previo a la pelea de Box contra Jermell Charlo USA, en la Explanada del Arena T-Mobile el 29 de septiembre de 2023. LAS VEGAS NEVADA ESTADOS UNIDOS
PUBLICATIONxNOTxINxMEXxCHNxRUS Copyright: xDavidxLeahx 20230929183529_BOX_2023_ALV_CHA_WEIGH_ALVAREZ44.

Graph facts:
- Floyd Mayweather -PARTICIPATED_IN-> Canelo’s fight with Mayweather
- Showtime Boxing to take final 10-count: Top 5 greatest fights revisited -MENTIONS-> Floyd Mayweather
- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled Lennox Lewis -MENTIONS-> Floyd Mayweather
- Floyd Mayweather -PARTICIPATED_IN-> Floyd Mayweather’s 2013 performance against Canelo Alvarez
- Floyd Mayweather -PARTICIPATED_IN-> Conor McGregor’s fight against Floyd Mayweather

Entities: Floyd Mayweather, net worth, billionaire status, Potential Conor McGregor Fight, Weigh-in ceremony, Boxing fight, T-Mobile Arena, Nevada, United
States, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status, Conor
McGregor, Jermell Charlo, Canelo Alvarez, Essentially Sports, Las Vegas
</item 7>

<item 8>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
entity: Sam Bankman-Fried
relationship: MENTIONS
related: Sam Bankman-Fried
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 8>

<item 9>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
So, a fight with McGregor could earn him anywhere from $55 to $60 million, adding to his $180 million. Meaning his total net worth would jump up to $240
million.

ADVERTISEMENT Article continues below this ad

While matching Floyd’s net worth is no easy feat, if anyone can do it – it’s Canelo Alvarez. However, only time will tell if the fight can ever materialize. In
case it does, it will be a spectacle the whole world would come together to witness. Do you think McGregor vs. Canelo can happen? Who would win the fight?

Watch This Story: Here’s How $200,000,000 Worth Canelo Alvarez Spent His First Ever Million

Graph facts:
- Floyd Mayweather -PARTICIPATED_IN-> Canelo’s fight with Mayweather
- Showtime Boxing to take final 10-count: Top 5 greatest fights revisited -MENTIONS-> Floyd Mayweather
- Top ten craziest fight endings in boxing history – Floyd Mayweather’s ruthless KO, Mike Tyson’s bite and a baffled Lennox Lewis -MENTIONS-> Floyd Mayweather
- Floyd Mayweather -PARTICIPATED_IN-> Floyd Mayweather’s 2013 performance against Canelo Alvarez
- Floyd Mayweather -PARTICIPATED_IN-> Conor McGregor’s fight against Floyd Mayweather

Entities: Floyd Mayweather, net worth, billionaire status, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd
Mayweather’s Net Worth and His Billionaire Status, Conor McGregor, McGregor vs. Canelo fight, Canelo Alvarez, Essentially Sports
</item 9>

<item 10>
Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass Floyd Mayweather’s Net Worth and His Billionaire Status (Essentially
Sports, 2023-10-18T21:27:46+00:00)
Crossover boxing matches are all the rage these days. Until a few days ago, the phenomenon was limited to influencers and MMA stars. But after a recent quarrel
on X (formerly Twitter) between Conor McGregor and Canelo Alvarez, a potential crossover boxing match has the boxing world buzzing with excitement. If the fight
were to turn into a reality, it could help ‘Cinnamon’ get closer to his former rival Floyd Mayweather in terms of net worth.

America’s Favorite Video Today

The talks of Conor McGregor vs. Floyd Mayweather began with a tweet that launched a fiery exchange of words between these two mammoths of combat sports. The
brash and confident ‘Notorious’ McGregor turned to X, praising Mayweather’s 2013 performance against Canelo. But he also boasted about landing more punches on
‘Money’ Mayweather than the Mexican superstar. Canelo quickly retorted, asserting he would need only one hand to make quick work of McGregor.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> X
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> X
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> X
- Is Google Search better than the rest? And is that fair? -MENTIONS-> X
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> X

Entities: X, Essential Sports, Floyd Mayweather, Potential crossover boxing match between Conor McGregor and Canelo Alvarez, Floyd Mayweather’s 2013 performance
against Canelo Alvarez, net worth, billionaire status, combat sports, Potential Conor McGregor Fight Could Help Canelo Alvarez Come a Step Closer to Surpass
Floyd Mayweather’s Net Worth and His Billionaire Status, Conor McGregor, Canelo Alvarez, boxing
</item 10>
=== END ITEM id=2d917aa729a71fab ===

=== ITEM id=0351dce5f7a9643f ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: Wang testified that Bankman-Fried said withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.1: Huang said Paradigm was told Bankman-Fried did not want investors as directors.
  Fact 1.2: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 2.2: The defense countered that the FTX founder acted in good faith.
  Fact 2.3: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.4: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, extensive media tour, interviews, Andrew Ross Sorkin, Good Morning
America, The New Yorker, FTX, Sam Bankman-Fried, The Verge
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Ellison, FTX, Wang, Sam Bankman-Fried, Binance, The Verge
</item 3>

<item 4>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.

Graph facts:
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> defense
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Bitcoin
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gisele Bündchen
- Israel's defense undone by reliance on technology -MENTIONS-> defense
- Patrick Kane destinations: Sabres, Red Wings among potential landing spots for free agent winger -MENTIONS-> defense

Entities: Gisele Bündchen, Bitcoin, defense, Elizabeth Holmes, cryptocurrency, Is Sam Bankman-Fried’s defense even trying to win?, Juilliard, major VC firms,
Elizabeth Holmes trial, FTX, Sam Bankman-Fried, The Verge
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto

Entities: crypto, Alameda, FTX trial, second trial, misappropriated FTX customer deposits, political contributions, The FTX trial is bigger than Sam
Bankman-Fried, real estate, FTX, Sam Bankman-Fried, The Verge
</item 5>

<item 6>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
connections: ['Josh Naftalis', 'Naftalis', 'Jacquelyn Melinek', 'Chain Reaction podcast', 'Apple Podcasts', 'bankruptcy', 'Chain Reaction newsletter', 'jury
selection', 'Robert Boroujerdi', 'Third Point', 'University of Notre Dame', 'Easton', 'FTX Ventures', 'Chinese officials', 'Paying Chinese officials $150
million to restore access to $1 billion in frozen trading accounts', 'charities', 'SBF Trial', 'FTX collapse', 'SBF', 'Matt Huang', 'Paradigm', 'class-action
lawsuit', 'defrauding customers', 'Chain Reaction', 'SBF trial’s jury', 'Gary Wang', 'Caroline Ellison', 'crypto', 'Thane Rehn', 'Mark Cohen', 'Alameda
Research', 'Alameda', 'bribery', 'TechCrunch', 'Equity', 'TechCrunch+', 'money laundering', 'political causes', 'Peter Easton', 'Can Sun', 'real estate', 'FTX',
'Sam Bankman-Fried', 'investments', 'Spotify', 'fraud', 'Stanford University']
passages: ['SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nSBF trial: Everything to know from
the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading crypto exchange FTX and sister
company Alameda. Here’s our latest coverage as the case nears its end.\n\nThe trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange
FTX, is entering its next chapter as proceedings resume Thursday. As confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a
handful of other witnesses as the onus shifts toward the defense after the prosecution brought its case to its conclusion.\n\nAs former federal prosecutor Josh
Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential sentencing for seven counts tied to fraud
and money laundering.\n\n“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.', 'SBF Trial: The latest updates from the FTX
collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\n“He can’t just walk out if he doesn’t like how it’s going.”\n\n\n\nNaftalis spoke at length
about what comes next in the trial, and addressed common misconceptions about the prosecution and defense in the latest episode of TechCrunch’s Chain Reaction
podcast, which you can listen to below, or on Apple Podcasts or Spotify.\n\nThe SBF trial is one of the biggest cases of its kind within the crypto space. Our
resident crypto expert Jacquelyn Melinek has been on the ground at the trial since it started, and continues to cover the ins and outs of the proceedings. But
the trial has been on pause from October 20 to today, so there’s plenty to catch up on if you need a refresher ahead of it entering its final few phases.\n\nFor
those in need of a total refresher on the case, her breakdown of how FTX went from the third-largest crypto exchange valued at a peak of $32 billion to
bankruptcy will get you up to speed.', 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nYou can
also follow along with her and the TechCrunch team’s coverage in the Chain Reaction newsletter, which drops Thursdays at 12 p.m. PT, and the Chain Reaction
podcast, which will hit your feed every Thursday.\n\nHow to follow the SBF trial\n\nThe trial kicked off with jury selection October 3. From there, access to
the proceedings is limited, as no devices are allowed; reports are being physically put together within the courtroom (or sent to overflow rooms).\n\nBeyond our
daily coverage, additional expertise and commentary will live on TechCrunch+. There, you can find stories like this in-depth breakdown of what to expect from
both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their arguments and what the takeaways were from the opening
arguments.\n\nSBF trial: What we learned in week three\n\nThe third week of the trial featured a couple of noteworthy moments for the prosecution, including
FTX’s former general counsel Can Sun.', 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nHe, like
many of SBF’s former colleagues, opted to cooperate with authorities and testified that SBF asked him to create a theoretical “legal justification” for the use
of billions in consumer funds just days before FTX filed for bankruptcy.\n\nRobert Boroujerdi, managing director of the hedge fund Third Point, testified about
his experience working with SBF and FTX, in a partnership that ultimately resulted in a $60 million investment and a lack of awareness about the special
relationship between Alameda Research and FTX. Had he known of such a relationship, he testified that Third Point would not have invested in FTX in the first
place.\n\nAnd earlier in the week, a University of Notre Dame professor who helped the government prosecution of Enron and WorldCom, Peter Easton, testified
about what he uncovered after tracing the flow of billions of dollars between Alameda Research and FTX.', 'SBF Trial: The latest updates from the FTX collapse’s
courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nMuch of said funding came from customers, and Easton testified that he found that user deposits were
used by both FTX and Alameda to purchase real estate, make investments or direct funding toward political causes and charities.\n\nSBF trial: What we learned in
week two\n\nThe second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from
customers to repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture
fund, FTX Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.\n\nAnd though there are no charges of bribery in
the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in frozen trading accounts.\n\nSBF trial: What we
learned in the week one\n\nAs expected, the opening statements in the SBF trial were spicy.', 'SBF Trial: The latest updates from the FTX collapse’s courtroom
drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nThe prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF
alongside rich and powerful figures to illustrate the peak of his power and influence prior to FTX’s implosion.\n\n“This man stole billions of dollars from
thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he emptied the accounts of ordinary
customers, too. He bought himself wealth, power and influence.”\n\nThe defense, on the other hand, depicted SBF and his colleagues as in over their heads,
without ill-will as FTX collapsed around them.\n\n“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said.
“They had to figure out how to navigate a world where they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of
their customers, managing hundreds of employees, all while building up their actual exchange.”\n\nRead a recap of the prosecution and defense’s full arguments
here.', 'SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nRead a recap of the prosecution and
defense’s full arguments here.\n\nFTX’s co-founder and CTO Gary Wang testified Thursday on the extent to which Alameda Research was used for whatever purposes
he and SBF saw fit. The crypto trading firm, according to Wang, pulled funding directly from FTX customers, whose transactions would be funneled toward Alameda
and then directed elsewhere.\n\nCheck here for our rundown of Wang’s testimony so far, resulting from a guilty plea in December 2022.\n\nThursday also brought
in testimony from Matt Huang, co-founder and managing partner at crypto investment firm Paradigm. That firm invested $278 million in FTX across 2021 and 2022,
and is part of a class-action lawsuit accusing it and others of defrauding customers by promoting FTX.\n\nHis testimony centered around being left uninformed
about FTX’s utilization of customer money to keep the Alameda Research afloat, a practice that Huang says would have halted his involvement.', 'SBF Trial: The
latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)\nThe full breakdown of his testimony on his relationship with
FTX, stretching back to 2019, can be found right here.\n\nWe also got a look at the composition of the SBF trial’s jury, which ranges from a train conductor to
a retired investment banker who attended Stanford University. You can listen to a special joint episode of our podcasts Equity and Chain Reaction to get a
better sense of the vibe between the jurors and the rest of the court.\n\nFor a full rundown of our coverage of the SBF trial, check below:']
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Gary Wang

Entities: Gary Wang, Securities and Exchange Commission, misappropriation of customer funds, Alameda, Christian Everdell, spot margin program, Sam Bankman-Fried
was a terrible boyfriend, FTX, Sam Bankman-Fried, trial, fraud, The Verge
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda, Sam Trabucco, MobileCoin loss, Singh, FTX, Wang, Sam Bankman-Fried,
MobileCoin, The Verge
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Sequoia Capital
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Sequoia Capital
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Sequoia Capital
- Matt Huang -WORKS_FOR-> Sequoia Capital
- Sequoia Capital -PARTICIPATED_IN-> OpenAI $300M share sale

Entities: Sequoia Capital, Alameda, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, Hermine Wong, lying to FTX investors, FTX, Sam Bankman-Fried,
Binance, SEC, Coinbase, The Verge
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.

Graph facts:
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> crypto
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Gary Wang
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> The New York Times
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Sequoia Capital

Entities: Gemini, fraud trial, The FTX trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, Kraken, Coinbase, crypto industry, The Verge
</item 10>
=== END ITEM id=0351dce5f7a9643f ===

=== ITEM id=3e64a186bdfb54a9 ===
Gold sentences and facts:
Sentence 0: Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.
  Fact 0.0: Zazie Beetz will be back as Sophie.
  Fact 0.1: Sophie is Arthur's neighbor.
  Fact 0.2: Sophie is Arthur's entirely unwitting love interest.
Sentence 1: But then, there’s MJ as well who has made a comeback from the first installment.
  Fact 1.0: MJ is also present in the story.
  Fact 1.1: MJ has made a comeback from the first installment.

Retrieved items:
<item 1>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Xbox
- Xbox -MENTIONS-> mobile market
- Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked -MENTIONS-> Xbox
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Xbox
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Xbox

Entities: Xbox, Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Starfield, Essentially
Sports, PlayStation, PlayStation 5
</item 1>

<item 2>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Wake Up, Good Men, Sable, Essentially
Sports
</item 2>

<item 3>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
There’s a visible growth in her arc especially considering her profession as a journalist. Yes, she’s Peter’s love interest, but that’s not everything she’s
there for. Players can easily see it happening throughout the game’s duration.

The verdict?

ADVERTISEMENT Article continues below this ad

It’s obvious to expect that a lot of fans who didn’t like MJ’s missions in the first installment won’t like her this time as well. But considering how she was
before and how she’s in Spider-Man 2, the overall change is quite praiseworthy. Things are different this time, thankfully.

MJ’s missions are now having some actual meaning attached to them rather than acting like fillers. Players who want to have a few interesting moments aside from
the main quests featuring Miles and Peter can have a sense of finishing well-designed stealth quests. Mary’s gameplay missions end up increasing the overall
quality of Spider-Man 2.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Xbox
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> The Daily Bugle
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Spider-Man

Entities: Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, journalist, Peter, Essentially Sports, Spider-Man 2, Miles
</item 3>

<item 4>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Marvel’s Spider-Man 2
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Marvel, Spider-Man 2 toys with Marvel history to be as much fun as possible, Chelsea, the Bronx, mass incarceration,
recidivism, reintegration into society, surveillance state, modern technology, release of Marvel’s Spider-Man 2, Polygon
</item 4>

<item 5>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Insomniac’s 2018 installment, Grand Central
Station, Mr. Negative, Mr. Negative’s demons, hunters, Wake Up quest, taser, stun gun, bomb, Mary Jane, Essentially Sports
</item 5>

<item 6>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Insomniac’s team, Peter, Essentially
Sports, Miles
</item 6>

<item 7>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, The Daily Bugle, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, PS5
</item 7>

<item 8>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, Peter
Parker, Miles Morales, PS5
</item 8>

<item 9>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2024

Joker: Folie à Deux: Oct. 4

Todd Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with ​​Joaquin Phoenix reprising his role as
Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest,
Sophie.

Graph facts:
- The future of DC movies in 2022 and beyond -MENTIONS-> Colin Farrell
- The future of DC movies in 2022 and beyond -MENTIONS-> Ryan Reynolds
- The best movies of 2023 so far -MENTIONS-> Joaquin Phoenix
- The future of DC movies in 2022 and beyond -MENTIONS-> Netflix
- The future of DC movies in 2022 and beyond -MENTIONS-> DC Comics

Entities: Todd Phillips, Joker, Joker: Folie à Deux, King of Comedy, Joaquin Phoenix, Arthur Fleck, Lady Gaga, Harley Quinn, Zazie Beetz, musical thriller, The
future of DC movies in 2022 and beyond, Martin Scorsese, Sophie, DC, Polygon
</item 9>

<item 10>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Spider-Man, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially
Sports
</item 10>
=== END ITEM id=3e64a186bdfb54a9 ===
