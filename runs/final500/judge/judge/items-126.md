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

=== ITEM id=0a9981d0d3f9e089 ===
Gold sentences and facts:
Sentence 0: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 0.0: Before his fall, Bankman-Fried presented himself as the Good Boy of crypto.
  Fact 0.1: Bankman-Fried presented himself as the trustworthy face of a sometimes-shady industry, the crypto industry.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: During cross-examination, Huang said.
  Fact 1.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.2: Huang said Paradigm was told that Bankman-Fried didn't want investors as directors.
  Fact 1.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 2: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 2.0: Huang said the decision was Bankman-Fried's.
  Fact 2.1: Huang said Bankman-Fried made the decision because he was the CEO of FTX.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 3.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 3.2: The defense argued that the FTX founder acted in good faith.
  Fact 3.3: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 3.4: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
He could face up to 115 years in jail if convicted on all charges.

The crypto industry as a whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global
Trading in January.

Where we are today

Bankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal prosecutor, will be the lead attorney.
If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial related to Jeffrey Epstein. He requested an
early release for Bankman-Fried but was denied.

With the trial beginning, we’re witnessing how FTX’s story ends. But what’s on our mind is what happens to the investors and creditors affected by the collapse?
And what happens to the billions in crypto assets tied up in legal proceedings?
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Assistant United States Attorney Nicolas Roos questions Gary Wang during Sam Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt cryptocurrency
exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

An $8 billion bug

Adam Yedidia, who was the prosecution's second witness on Wednesday, continued his testimony on Thursday. Yedidia met Bankman-Fried in college at MIT, and the
pair remained close friends. Yedidia, assuming a robotic posture on the stand, worked out of FTX's Hong Kong office from January to October of 2021 and then in
the Bahamas until last year's collapse. In his testimony, he referred to a group Signal thread called "People of the House," referring to Bankman-Fried's $35
million penthouse, where many employees lived.
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 3>

<item 4>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 7>

<item 8>
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
</item 8>

<item 9>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
As a result of today’s testimony, I am both more and less confused. I am clearer on what probably happened at FTX. What I don’t understand is why Gary Wang did
it.

Wang was the co-founder and chief technology officer of FTX, as well as the co-founder of Alameda Research. He owned only 10 percent of Alameda and 17 percent
of FTX. He got a $200,000 a year salary, with no performance bonuses. (Okay, sure, at times he was a billionaire on paper.) By contrast, Bankman-Fried owned 90
percent of Alameda and 65 percent of FTX.

Yet Wang’s testimony suggests he took bizarre risks for his own relatively small share. Bankman-Fried didn’t code, so none of his fingerprints are on the
technical changes that let FTX allegedly defraud its customers and investors. Meanwhile, Wang supervised the code behind a key prosecution claim: that Alameda
Research was afforded secret, special privileges with FTX’s funds.
</item 9>

<item 10>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
"FTX defrauded all its customers," he said.

Matt Huang, co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is
charged with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via
Getty Images Yuki Iwamura | Bloomberg | Getty Images

Investment to zero

The third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto venture capital firm that invested over $275 million
in FTX. That stake was wiped out. Huang testified about his firm's due diligence on FTX, and he told the court that Bankman-Fried assured him that funds would
be used for FTX and not Alameda.
</item 10>
=== END ITEM id=0a9981d0d3f9e089 ===

=== ITEM id=9e58e86a039d7f57 ===
Gold sentences and facts:
Sentence 0: Ketamine is primarily used as an anesthetic during surgical procedures — but in recent years it's been used as a remedy for treatment-resistant depression .
  Fact 0.0: Ketamine is primarily used as an anesthetic during surgical procedures.
  Fact 0.1: In recent years, ketamine has been used as a remedy for treatment-resistant depression.
Sentence 1: If it worked with the fish, they might then be able to figure out how the diet effects behavior and improve the effectiveness of the treatment for humans.
  Fact 1.0: If the treatment worked with fish, researchers might then be able to figure out how the diet affects behavior.
  Fact 1.1: If the treatment worked with fish, researchers might then be able to improve the effectiveness of the treatment for humans.

Retrieved items:
<item 1>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"To use ketamine, it is important to establish a clear diagnosis of severe depression or anxiety and/or post-traumatic stress disorder," he told Fox News
Digital. "This requires an expert and cannot be self-diagnosed properly."

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction."

Those who think they may benefit from ketamine treatment should consult a doctor, Johnson advised.

"Ketamine is generally safe when administered by trained health professionals in a medical setting," Johnson said.

"In those settings, it is best delivered by the IV route, where the dose can be calculated accurately."

When purchasing ketamine outside a doctor’s prescription, there is also the risk of receiving a tainted product, experts warned. While people might assume
they're purchasing straight ketamine, the drug is often mixed with stimulants like cocaine or phencyclidine (PCP), Johnson noted.
</item 1>

<item 2>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ketones are known to create detectable increases in gene expression in cells. Pulling apart exactly how things like, diet, environment, genes and
neurotransmitters are linked is incredibly difficult but could reveal which pathways are best to target for autism treatments or could identify a specific
ketone which works more efficiently than others.

While the cavefish are not an exact model for human autism this work shows how these experiments shine light on the biology and evolution of these behavioral
conditions and offer new directions for researchers to follow.

Reference: Motoko Iwashita, et al. Metabolic shift toward ketosis in asocial cavefish increases social-like affinity, BMC Biology (2023), DOI:
10.1186/s12915-023-01725-9

Feature image credit: Masato Yoshizawa

Correction: This article was edited on November 2, 2023. Ryan Lee was originally described as working at the University of Hawai‘i Mānoa.
</item 2>

<item 3>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Surprisingly effective

Again, Yoshizawa was a bit skeptical. “I doubted it at first,” he said. “I didn’t think it was going to work, but it did.”

In the experiment, cavefish where fed the same ketogenic milk provided to human patients, albeit with a few modifications for fish consumption, and their
behavior was monitored. As a comparison, a type of A. mexicanus fish that lives in rivers and not caves were also tested.

The surface fish do not display the same autism like behaviors as their cave dwelling relatives. In the presence of other surface fish, individuals will begin
to follow each other and swim together, something rarely seen in cavefish, Yoshizawa said. The surface fish also do not do the repetitive behavior of swimming
in circles.

Using these fish as a comparison, Yoshizawa and his students watched and waited. Amazingly, after a month of the ketogenic diet, the cavefish began to act like
the more social surface fish.
</item 3>

<item 4>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Buprenorphine, a drug that is prescribed to treat opioid use disorder , was in Perry’s system at the time of his death.

Guidelines to be published soon

ASKP3 announced that it will soon publish guidelines for safe use of the drug.

"In the wake of Matthew Perry’s autopsy report, we are committed to creating and publishing guidelines for at-home ketamine use, which should only be practiced
with a very specific patient profile and only in the context of a close relationship with a physician and in-office follow-up," the organization said in a
released statement.

CLICK HERE TO GET THE FOX NEWS APP

It is important for people to recognize the distinction between different uses of the drug, according to Prashad.

"In reality, the ways in which ketamine is used for anesthesia versus depression versus abuse are all very different in terms of dosage and frequency."
</item 4>

<item 5>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Johnson, however, noted that ketamine is a "moderately addictive substance."

He told Fox News Digital, "Even long-term users typically only develop a behavioral or psychological dependence. Physical dependence can occur, but usually the
withdrawal signs are mild to moderate."

Dr. Sandhya Prashad, president of the American Society Of Ketamine Physicians, Psychotherapists and Practitioners (ASKP3) in Florida, said there is "an
extremely low risk of addiction, craving or withdrawal" when ketamine is used in low doses in a monitored clinical setting.

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction ," she told Fox
News Digital.

Myth No 3: Ketamine can be self-prescribed and purchased online

Ketamine should only be obtained and administered by a licensed and experienced medical doctor, Johnson said.
</item 5>

<item 6>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
ULTRAPROCESSED FOODS WITH ‘FEEL-GOOD CHEMICALS’ COULD BE AS ADDICTIVE AS CIGARETTES AND DRUGS, STUDY SUGGESTS

"These can produce strong cardiac effects of a very high blood pressure and pulse, and can lead to serious medical complications," he said.

Myth No. 4: Intranasal or micro-dosing is safe for recreational use

While using small doses of ketamine may seem safer than the larger doses given via IV, Johnson warned that when this drug is self-administered, there is a
danger of exceeding safe limits.

"The total cumulative dose could end up being higher than a standard dose of a medically administered intravenous dose," he told Fox News Digital.

Intranasal use can also lead to "more aggressive drug-seeking" behaviors, Johnson warned, as the "high" of taking the ketamine is paired with cues in the user’s
environment — which are not present in a medical office.
</item 6>

<item 7>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
The fish avoid their peers and engage in repetitive behavior, such as swimming in circles.

“Many people first doubted that the fish have an autism-like state; I also doubted it at first,” said Yoshizawa. But as he soon found out, even patterns of gene
regulation resembled autistic patients.

Believing that cavefish might be useful for the study of autism, Yoshizawa approached his colleague Ryan Lee, a physician working with autism patients at
Shriners Hospital for Children Honolulu. “When I showed him the video of the cavefish he immediately said, ‘yes, that is very similar to what I observe in my
patients’,” recounted Yoshizawa.

Lee, who has experimented with ketogenic diets to improve autism symptoms with some success suggested they try it with the fish. In humans, not all patients
receiving the ketogenic diet improve, but some do. If it worked with the fish, they might then be able to figure out how the diet effects behavior and improve
the effectiveness of the treatment for humans.
</item 7>

<item 8>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
They would follow each other in groups and ceased going round in circles. There were some other behaviors, such as attention to a specific task and sleeping,
that were unaffected, but overall the results were promising and according to Yoshizawa, suggest dopamine could be key to how the diet affects behavior.

“We suspect that due to this behavior variation, probably the target of the ketogenic diet is the dopaminergic system,” he said. The challenge now is teasing
out all the complex factors that may be involved.

Diet, environment, and genes – a complex recipe

According to Yoshizawa, there are two plausible ideas as to how the ketones produced by a ketogenic diet are acting on behavior. The first involves the
mitochondria, which use either carbs or fat to produce energy in our cells, and the other involves epigenetics, which simple refers to any non-genetic influence
which turns genes on and off.

Ketones are known to create detectable increases in gene expression in cells.
</item 8>

<item 9>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Myth No. 1: Ketamine is a new treatment

In reality, the drug has been used for depression, post-traumatic stress disorder and pain for more than 10 years, according to Dr. Patrick Sullivan, medical
director of Initia Nova Medical Solutions in New Jersey .

The FDA approved ketamine for anesthesia in both humans and animals since 1970, the doctor noted.

After successful clinical trials , the FDA approved the drug in 2019 as a medication for treatment-resistant depression.

OZEMPIC AND WEGOVY WEIGHT LOSS DRUGS COULD HELP REDUCE ALCOHOL USE DISORDER SYMPTOMS, STUDY SUGGESTS

"There are hundreds of studies showing it to be safe and effective in the outpatient setting," Sullivan told Fox News Digital.

Myth No. 2: Ketamine is powerfully addictive

Ketamine is "legally recognized as having the potential for abuse and for both psychological and physical dependence," according to the American Addiction
Centers website.
</item 9>

<item 10>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
When thinking about treatments for conditions like autism spectrum disorder, diet is likely not high on the list. But clues from an eyeless species of
cave-dwelling fish point to a link between a trendy weight loss diet, the brain, social behavior, and autism.

The ketogenic diet is designed to induce a physiological state called ketosis. This happens when the body no longer has enough carbohydrates to burn for energy
and switches to fat as an energy source. Breaking down fat for energy produces a group of molecules called ketones, which travel throughout the body.

High in protein and fat but low in carbs, the ketogenic diet is popular for its weight loss effects but also appears to have some neuroprotective and
anti-inflammatory effects too. Children with epilepsy are sometimes placed on this diet because it helps reduce the frequency of seizures.

The exact mechanism behind these neurological effects is still a mystery and there have been reports of long-term side effects in people trying the diet outside
of clinical settings.
</item 10>
=== END ITEM id=9e58e86a039d7f57 ===

=== ITEM id=e7c2cee3f66d0924 ===
Gold sentences and facts:
Sentence 0: Additionally, Caesars has a competitive welcome bonus for new players and runs NBA betting promos for existing players.
  Fact 0.0: Caesars has a competitive welcome bonus for new players.
  Fact 0.1: Caesars runs NBA betting promos for existing players.
Sentence 1: New customers can get up to $1000 in bonus bets if they lose their first bet.
  Fact 1.0: New customers can get up to $1000 in bonus bets.
  Fact 1.1: New customers can get bonus bets if they lose their first bet.
Sentence 2: It has a wide range of bets for the PGA Tour and other major events, such as winning margin, top finishes, and live betting props.
  Fact 2.0: Caesars has a wide range of bets for the PGA Tour.
  Fact 2.1: Caesars has a wide range of bets for other major events.
  Fact 2.2: The bets include winning margin bets.
  Fact 2.3: The bets include top finish bets.
  Fact 2.4: The bets include live betting props.
Sentence 3: We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.
  Fact 3.0: The writer expects Caesars to offer in VT the same offer it has elsewhere.
  Fact 3.1: New sign-ups are eligible to receive up to a $1,000 Bonus Bet.
  Fact 3.2: New sign-ups are eligible if their first bet loses.

Retrieved items:
<item 1>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Time limits

These would force you to use a bonus before an expiration date or meet all wagering requirements within a specific timeframe.

Maximum withdrawals

Some offers could limit your ability to withdraw winnings of a particular amount, forcing you to make multiple withdrawal requests on big wins.

Vermont sport-specific betting promos

Vermont has no teams in the four major US pro sports leagues. However, state residents have been known to support Boston-based teams, like the Red Sox (MLB),
the Celtics (NBA), the Bruins (NHL), and the Patriots (NFL).

Because of those teams’ proximity to the state, you can expect Vermont online sportsbooks to run related promos (eg. NFL betting promos) to attract fans.
College sports are big, too, which means several betting promos should also surround local college teams.

How do I claim a Vermont sportsbook bonus?

Many times, all you’ll need to do is sign up for a new account through the links on this page to claim a bonus from a Vermont online sportsbook.
</item 1>

<item 2>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
You can read more about FanDuel Sportsbook Vermont's prelive offer here.

You can bet on around two dozen different sports at FanDuel. The site offers lots of props, futures, alternate lines, and popular premade same-game parlays.
There are lots of ongoing promos, too, like profit boosts, free pools, pick-em contest, special “no sweat” bets and more.

Here are more details of that FanDuel Sportsbook prelive & welcome offers:

Use the links on this page to create your FanDuel Sportsbook account.

to create your FanDuel Sportsbook account. Register before Jan. 10, 2024 at 11:59 p.m. ET and receive $100 in Bonus Bets available to use on the Jan. 11 launch
day

and receive available to use on the Jan. 11 launch day Once the site goes live, make a first deposit of at least $10 into your FanDuel account and place a
qualifying bet of at least $5 .
</item 2>

<item 3>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Those signing up at FanDuel Sportsbook Vermont by Jan. 10, 2024 will find $100 in Bonus Bets in their accounts when the site launches on Jan. 11. In addition,
FanDuel Sportsbook VT will also feature the same welcome offer it has in other states, inviting new sign-ups to bet at least $5 and get $200 in Bonus Bets. Be
aware that sometimes FanDuel add perks to that offer, such as they did at the start of the NBA season when they threw in a free three-month subscription to NBA
League Pass.

What can I win with the FanDuel Sportsbook welcome offer?

With the FanDuel offer, whatever you win with your Bonus Bets is yours to keep. If you can make some successful bets, that should mean a nice profit given the
fact that with the prelive offer you received $100 Bonus Bets for free, and with the welcome offer you only have to make a $5 bet up front in order to receive
the $200 in Bonus Bets.
</item 3>

<item 4>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
DraftKings also has a prelive offer already available to Vermont users. Be aware, though, that those who accept the prelive offer will not be eligible for
DraftKings' welcome offer later on.

If you use our links and sign up for a DraftKings Sportsbook VT account between now and the moment the site goes live on Jan. 11, you will receive $200 in Bonus
Bets, issued as eight $25 Bonus Bets. You'll have to wager these (i.e., you can't withdraw them), and you must do so within 7 days or they'll expire.

If you choose to wait and instead take the welcome offer, you'll find DraftKings Sportsbook actually has a multi-part offer worth up to $1,200. The offer
includes a deposit match bonus worth up to $1,000, a $50 Bonus Bet just for depositing at least $5, and another $150 in Bonus Bets for making a first wager of
at least $5.
</item 4>

<item 5>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
11, 2024. BetMGM and ESPN Bet are also among the applicants, making them possible candidates to launch in Vermont in the new year as well. Here are details
regarding those five online sportsbooks and their welcome offers.

FanDuel Vermont promo code

Vermont already knows FanDuel thanks to its daily fantasy sports site. FanDuel Sportsbook is one of the most popular across the US, and FanDuel has already
applied for a license to launch in Vermont. FanDuel has a great welcome offer inviting new sign-ups simply to make an initial $5 wager and get back $200 in
Bonus Bets.

Meanwhile, for Vermont bettors FanDuel is running an Early Sign Up Special that gets you an extra $100 in Bonus Bets for signing up early between now and Jan.
10, 2024. That means you can receive a total of $300 in Bonus Bets to get you started at FanDuel Sportsbook Vermont.
</item 5>

<item 6>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Sometimes, you may need to enter the promo or bonus code, but those will all be listed on this page as well.

You may need to place a qualifying bet or meet specific wagering requirements to get your hands on the bonus. All of this will be outlined in the terms and
conditions attached to any Vermont online sportsbook bonus offer.

Vermont sportsbook promo codes FAQ

Can I get more than one bonus from a Vermont online sportsbook?

No, you’ll almost certainly be limited to one bonus from each Vermont online sportsbook, and sometimes the limit will be one per household. However, you can
claim that one bonus from every Vermont online sportsbook. Sign up at as many sportsbooks as you like.

What sports can these bonuses be used on?

Pretty much any sport should qualify. If betting on the sport is available in Vermont, you can use your bonus on it.

Why should I use the promo codes here on The Sporting News?

The promo codes on this page get you the best bonuses available.
</item 6>

<item 7>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
There’s also the BetMGM Rewards program for added benefits.

Here’s how that welcome offer works at BetMGM Sportsbook:

Start by entering the bonus code SPORTSPICK when you create your BetMGM Sportsbook account.

when you create your BetMGM Sportsbook account. Deposit at least $10 .

. Make a first bet and if you lose get back the equivalent amount in Bonus Bets up to $1,500 within 24 hours of the bet being settled.

within 24 hours of the bet being settled. For losing bets up to $50, you receive the Bonus Bet as a single bet . If you bet more than $50 and lose, you’ll get
your refund (up to $1,500) in five Bonus Bets divided equally (20% each).

. If you bet more than $50 and lose, you’ll get your refund (up to $1,500) in divided equally (20% each). You’ll have seven days to use your Bonus Bet(s).

to use your Bonus Bet(s).
</item 7>

<item 8>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
. Deposit at least $10 into your account.

into your account. Make a first bet, and if you lose get the bet amount back as a Bonus Bet worth up to $500 . (Note that this amount varies from state to state
and could be different in Vermont.)

. (Note that this amount varies from state to state and could be different in Vermont.) You’ll see the Bonus Bet in your account within 24 hours , and then
you’ll have 30 days to use it .

, and then you’ll have . The Bonus Bet has to be used on wagers with odds of -200 or longer, and if you win with it you’ll receive the winnings but not the bet
amount back.

Vermont sportsbook bonuses pros and cons

Vermont Sportsbook Bonus Pros Vermont Sportsbook Bonus Cons Thousands of dollars in value will be up for grabs. Refunds are site credit or bonus bets, not cash.
Bonuses can be claimed easily via links or promo codes. Playthrough requirements are attached to site credit and bonus bets.
</item 8>

<item 9>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
This won’t be available on the sportsbook website.

Special event offers

Get a bonus when you place a particular bet on special event odds like World Series odds, Super Bowl betting lines, or March Madness. Special event bonuses are
often bonus bets or site credit. That means wagering requirements are attached, and payouts don't include the stake.

Loyalty offers and programs

Earn points with every bet, and the points are redeemable for rewards like cash and bonus bets.

Important Vermont sportsbook bonus and promo conditions

Terms and conditions will be attached to every bonus and promo offer from Vermont online sports betting sites. Be ready to see these stipulations.

Banking options

Your ability to withdraw the winnings from a bonus could be limited to specific banking methods.

Wagering requirements

You might be forced to bet a bonus amount a minimum number of times before you can withdraw winnings.

Betting limitations

There may be limits on the odds or bet types available for wagering bonus funds.
</item 9>

<item 10>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Playthrough requirements are attached to site credit and bonus bets. Bonuses can be used to bet on almost any sport. You won't earn back the stake on winning
bonus bets or bets with site credit.

Types of Vermont sportsbook promo and bonus offers

The welcome bonuses and ongoing promotions coming soon from new online betting sites in Vermont will fall into the following categories.

Deposit match bonuses

Get up to 100% of your first deposit matched in bonus funds. Sportsbooks will limit the bonus size you can claim and attach wagering requirements.

Referral bonuses

You share a personal referral code with friends and family, and if they use it to sign up, deposit, and bet, you both get a bonus. Sportsbooks will attach
wagering requirements to the bonus.

First-bet offers

You get a refund if you lose your first bet. Sportsbooks will limit the size of the refund you can claim. Most refunds are bonus bets or site credit, not cash.
That means wagering requirements are attached, and payouts don't include the stake.
</item 10>
=== END ITEM id=e7c2cee3f66d0924 ===

=== ITEM id=f7630e9cbfff60a1 ===
Gold sentences and facts:
Sentence 0: This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google’s on the road to victory.
  Fact 0.0: The whole line of argument is called 'market definition.'
  Fact 0.1: If the court decides the relevant market is phones and app stores, rather than specifically Android ones, Google is on the road to victory.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 4>

<item 5>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 8>

<item 9>
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
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 10>
=== END ITEM id=f7630e9cbfff60a1 ===

=== ITEM id=144d48f0fccd251e ===
Gold sentences and facts:
Sentence 0: Doe's story was adapted into the 1971 book "The Exorcist" by William Peter Blatty, which was then made into the 1973 movie starring Ellen Burstyn and Linda Blair.
  Fact 0.0: Doe's story was adapted into the 1971 book The Exorcist.
  Fact 0.1: William Peter Blatty wrote the 1971 book The Exorcist.
  Fact 0.2: The 1971 book The Exorcist was made into the 1973 movie.
  Fact 0.3: The 1973 movie starred Ellen Burstyn.
  Fact 0.4: The 1973 movie starred Linda Blair.
Sentence 1: Clive Barker’s 1987 directorial debut adapts his 1986 novella The Hellbound Heart to tell the story of Larry (Andrew Robinson) and Julia Cotton (Clare Higgins).
  Fact 1.0: Clive Barker's 1987 film is his directorial debut.
  Fact 1.1: Clive Barker's 1987 directorial debut adapts the 1986 novella The Hellbound Heart.
  Fact 1.2: Clive Barker wrote the 1986 novella The Hellbound Heart.
  Fact 1.3: The 1987 film tells the story of Larry, played by Andrew Robinson.
  Fact 1.4: The 1987 film tells the story of Julia Cotton, played by Clare Higgins.

Retrieved items:
<item 1>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Very loosely based on Argento’s own experience directing a failed production of Verdi’s Macbeth, Opera is a sumptuous and suspenseful slasher that holds the
viewer’s attention and absolutely refuses to let go. Like Betty, one feels compelled to watch this macabre display of ritualistic horror play out to its
conclusion in a desperate search of answers. Fortunately, unlike Betty, you won’t have to worry about losing your eyes if you find yourself overwhelmed by the
frights. —TE

Oct. 29: House of 1000 Corpses

Where to watch: AMC Plus, Pluto TV, Tubi

House of 1000 Corpses may be the true answer to the question “What is the best The Texas Chain Saw Massacre sequel?”

The movie follows a group of teens who are putting together a book on roadside attractions when they stumble upon a serial killer-themed rest stop run by a
weirdo. After their strange encounter, the kids set off on the road, only for their car to break down a few miles later.
</item 1>

<item 2>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.
</item 2>

<item 3>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.
</item 3>

<item 4>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Lovecraft and the anti-consumerist allegories of George A. Romero. The horrors hold up over 50 years after it was first released, and with just the right
combination of inadvertent silliness and legitimate terror, Messiah of Evil is a satisfying watch and an excellent start to a month of horrors. —Toussaint Egan

Oct. 2: Ginger Snaps

Where to watch: Criterion Channel, Shudder, Peacock, Vudu, Tubi, Freevee, Plex

There’s a sizable hole in the monster movie canon: There just aren’t enough good werewolf movies! Ginger Snaps is here to fix that. And if you love Jennifer’s
Body, you’re in for a bloody treat.

Directed by future Orphan Black co-creator John Fawcett, Ginger Snaps is a delightfully gruesome story about two sisters. One of them has gotten her period for
the first time, and is turned into a werewolf shortly after. Things get messy, fast.
</item 4>

<item 5>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
After inadvertently being resurrected by a drop of blood spilled by Larry on the floor of the house’s attic, Frank seduces Julia into luring new men to the
house so that he can drain their life force and fully regain his mortal form. Surrounding this core narrative is the the story of the Lament Configuration, a
puzzle box Frank acquired before his untimely death. When solved, it conjures hellish beings known as Cenobites to the mortal plane of existence, which indulge
in hellish exercises of sadomasochistic mutilation. Easily the best and most enduring of the Hellraiser movie series, Barker’s 1987 original is a must-watch for
horror fans. —TE

Hereditary

Director: Ari Aster

Cast: Toni Collette, Alex Wolff, Milly Shapiro

Where to watch: Max

Hereditary is a victim of its own success.
</item 5>

<item 6>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

If there’s one thing we love here at Polygon, it’s the Halloween season.

We cover horror year-round, whether it’s the best horror movies you can stream at home or the best horror movies on Netflix, and generally, we have our finger
on the undead pulse of the newest and ghastliest releases in horror. We even have a list of the best horror movies of the year (ranked by scariness, of course).

Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.
</item 6>

<item 7>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—PV

Oct. 31: The Exorcist III

Where to watch: Prime Video, Criterion Channel

Nearly 50 years since it was first released in theaters, William Friedkin’s The Exorcist has withstood the test of time as one of the most terrifying and
influential horror films in American cinema. The sequel that came closest to capturing that same lightning in a bottle is The Exorcist III, written and directed
by William Peter Blatty, the original screenwriter of the first film and author of the novel on which it was based.

When a series of gruesome murders matching the modus operandi of an infamous serial killer begins to terrorize the city of Georgetown, Lt. William F. Kinderman
(George C. Scott) investigates the psych ward of a local hospital in search of leads. Confronted with a suspect whose appearance matches that of his late friend
Father Damien Karras, Kinderman finds himself thrust into a battle for his very soul as he confronts an evil beyond his comprehension.
</item 7>

<item 8>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.

For the past three years, Polygon has put together a Halloween countdown calendar, selecting 31 of our staff’s top horror-themed or Halloween-adjacent picks
across movies and TV throughout the month of October, all available to watch at home. We’ve loved doing it, so much so that we’re bringing it back again — this
time with a whole new batch of films and shows to choose from.

Every day for the month of October, we’ll add a new recommendation to this countdown and tell you where you can watch it. So curl up on the couch, dim the
lights, and grab some popcorn for a terrifying and entertaining marathon of horrific delights.

Oct. 1: Messiah of Evil

Where to watch: Prime Video, Shudder, Pluto TV, Plex

A undersung classic of ’70s cosmic horror, Willard Huyck and Gloria Katz’s Messiah of Evil is a surreal and nightmarish experience that more than deserves its
status as a cult horror classic.
</item 8>

<item 9>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
The possessed-animatronics horror video game Five Nights at Freddy’s is less than a decade old, but a movie version has been in the works almost since the game
launched, which seems especially protracted for a development process. Warner Bros. bought the movie rights in 2015, and the project moved over to Blumhouse a
couple of years later. Game creator Scott Cawthon wrote and rewrote the script, while kid-horror filmmakers like Gil Kenan (Monster House) and Chris Columbus
(Gremlins, the first Harry Potter and Percy Jackson movies) signed on and jumped off the project.

Finally, under director Emma Tammi, the film is hitting theaters simultaneously with a Peacock streaming debut. The franchise’s most hardcore fans probably
consider themselves beyond ready to watch the Freddy’s film. But what about everyone else? If you’re not sure, have a look at this handy (and extremely
spoiler-light) Q&A.
</item 9>

<item 10>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE
</item 10>
=== END ITEM id=144d48f0fccd251e ===
