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

This file has 6 items.

=== ITEM id=260121021679d791 ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.0: On October 12, 2023, Israel warned 1.1 million Gaza residents to leave for the southern region.
  Fact 0.1: The warned Gaza residents were in the northern section of the enclave.
  Fact 0.2: Israel issued the warning in advance of a potential ground invasion.
Sentence 1: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 1.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 1>

<item 2>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 2>

<item 3>
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
</item 3>

<item 4>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
International aid groups are warning that they cannot deliver food and other basic services to people in the Gaza Strip and that a “dire” humanitarian crisis is
set to worsen.

International aid groups provide food and other means of support to about 63% of people in Gaza.

Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground
invasion.

On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential
ground invasion.

I am a scholar of peace and conflict economics and a former World Bank consultant, including during the 2014 war between Hamas and Israel.
</item 4>

<item 5>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 5>

<item 6>
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
</item 6>

<item 7>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 7>

<item 8>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 8>

<item 9>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 9>

<item 10>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
"Despite the clear threat posed by Gaza, the decision makers preferred to avoid a policy that would mean returning a large part of the country's population to
the reality of air raids and disrupted daily routines. The bottom line: the Iron Dome did not change the essential strategic situation; as firepower from the
Gaza Strip developed, so did the terrorist organizations’ ability to control daily life in Israel’s hinterland, hold it hostage, and restrain Israel from taking
proactive measures against them."

"Why didn't the relative success of the Iron Dome translate into the even more important strategic achievement of restoring Israel's freedom of action against
Hamas in the Strip?" asks Ortal rhetorically, and names several factors. "The first, like any other operational system, Iron Dome is also not perfect. The enemy
studies it and develops its own challenges against it, and the decision makers recognized the fact that the State of Israel is not hermetically protected.
Second, the enormous cost involved in deploying and operating [Iron Dome] batteries significantly impedes a policy based on continuous rounds of violence.
</item 10>
=== END ITEM id=260121021679d791 ===

=== ITEM id=e912afffe4fba062 ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker said, 'Hopefully we can go one better this year.'
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by ex-Collingwood player Chloe Molloy.
  Fact 1.2: The Swans have been competitive this season.

Retrieved items:
<item 1>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Yet the 22-year-old proved absolutely nerveless, firing down an ace to save the third of the three match points, and then immediately turning the tables by
breaking Djokovic in the very next game amid wild Italian celebrations in the stands at the Palacio de deportes Martin Carpena.

Sinner then served out for a famous win, ending Djokovic’s cherished unbeaten 21-singles winning streak in the Davis Cup and becoming the first man since 2019
to beat the Serb in singles twice in the same year. The drama between the two rivals wasn’t over, as the two teams inevitably both drafted in their top singles
players into the doubles decider. Sinner was again magnificent and still looking ominously fresh after more than four hours on court, with Djokovic’s weak
backhand into the net off the Italian’s second serve in the doubles on match point seeming to sum up his disappointment. The beneficiaries of all this?
</item 1>

<item 2>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.
</item 2>

<item 3>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Following their 2-0 victory over Finland in Friday’s semi-final, executed superbly without a dropped set in singles wins for Alexei Popyrin and Alex de Minaur,
team captain Hewitt said he couldn’t be prouder of their achievement in making back-to-back finals.

And even though they’ll be facing either a Serbian side led by world No.1 Novak Djokovic or an Italian team inspired by young superstar Jannik Sinner in
Sunday’s (Monday AEDT) final, the inspirational de Minaur was adamant: “No one really wants to play us.” He delivered once more for Australia, taking an 11th
victory in his last 13 cup singles matches, sealing Friday’s (Saturday AEDT) win by defeating Emil Ruusuvuori 6-4, 6-3. Alex de Minaur stretches to return the
ball against Finland’s Emil Ruusuvuori.
</item 3>

<item 4>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 4>

<item 5>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 5>

<item 6>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
</item 6>

<item 7>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.
</item 7>

<item 8>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 8>

<item 9>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
“Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022. De Minaur had been the hero on Wednesday, pulling out an
extraordinary comeback win over Jiri Lehecka with Australia on the verge of defeat in their quarter-final with the Czech Republic.

But he was happy to pass the laurels to Popyrin, a late squad replacement who’d been asked to take on the troublesome “second singles” role by Hewitt. Loading
“Alexei hasn’t played for Australia in a very long time [two years], had a massive match today against a very high-quality opponent and dug super deep,” said de
Minaur. Popyrin managed to deliver a rock solid game, making only 11 unforced errors to his erratic opponent’s 25, to prevail in just under an hour-and-a-half.
“It was probably the biggest win of my career,” he admitted.
</item 9>

<item 10>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with Lorenzo Sonego and spray massive forehand winners around liberally as they
outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic had three match points in the singles rubber. Credit: Getty Images It was a
crushing end to Djokovic’s hopes of ending his triumphant season, which has featured three grand slam victories and an ATP finals showdown victory over Sinner
just six days ago, with a second Davis Cup title. After telling fans to “shut up” earlier in the week, he again got irritated with the crowd in the doubles,
pretending to be the conductor to the Italian fans’ deafening chorus, but though he may be the greatest player who ever picked up a racquet, doubles looks an
alien game to him.
</item 10>
=== END ITEM id=e912afffe4fba062 ===

=== ITEM id=ec3e52e4f4b19751 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.1: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.2: Caroline Ellison was the CEO of Alameda.
  Fact 0.3: Bankman-Fried had 'secret access' to customer money, both cash and crypto.
  Fact 0.4: The government claimed that Bankman-Fried had this secret access to customer money.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: The decision was Bankman-Fried's, according to a female speaker.
  Fact 1.1: Bankman-Fried was the CEO of FTX.
  Fact 1.2: According to a female speaker, Bankman-Fried made the decision because he was the CEO of FTX.

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
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 3>

<item 4>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 5>

<item 6>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 10>
=== END ITEM id=ec3e52e4f4b19751 ===

=== ITEM id=375c26e09d1b9384 ===
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
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 1>

<item 2>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.
</item 2>

<item 3>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 3>

<item 4>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 4>

<item 5>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.
</item 5>

<item 6>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
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
</item 7>

<item 8>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 8>

<item 9>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.
</item 9>

<item 10>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 10>
=== END ITEM id=375c26e09d1b9384 ===

=== ITEM id=8ca928b0738990f5 ===
Gold sentences and facts:
Sentence 0: It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and leaders that will call time on their international careers at week’s end.
  Fact 0.0: It would seem that the All Blacks are playing for themselves as much as anyone.
  Fact 0.1: The Springboks play for their country.
  Fact 0.2: A generation of players and leaders will call time on their international careers at week's end.
Sentence 1: The Springboks took a 12-3 lead after 34 minutes when the All Blacks once again failed to release on their goal line.
  Fact 1.0: The Springboks took a 12-3 lead.
  Fact 1.1: The Springboks took the 12-3 lead after 34 minutes.
  Fact 1.2: The 12-3 lead came when the All Blacks once again failed to release on their goal line.
Sentence 2: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 2.0: The preceding situation (named 'It' in this sentence) comes despite Argentina knocking over the All Blacks last year in Christchurch.
  Fact 2.1: The preceding situation also comes despite a maiden defeat in 2020 in Sydney.
  Fact 2.2: There was a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.
</item 1>

<item 2>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“There were a few tears and I think that’s awesome to see, players showing vulnerability, just opening up about how much this tour and this sort of game means.



“There were some (people) saying this is probably one of the most important games of All Blacks history. (The) All Blacks went through a bit of a hard path to
get here, media, friends, family even criticising us and all that, and we stuck together. We weren’t even said to be in the final and now look at us, we’re
here.

Advertisement

“Different players take it differently and some of the older players probably were hit a little bit harder and it was good to see just a bit of emotion and the
want from them. Seeing that gives all the other players an extra two, three per cent to think ‘OK, we are going to the well and we’ve got to give it our all’.”

The All Blacks will experience a changing of the guard next season.
</item 2>

<item 3>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Pressure? The South African laughs at the notion of what pressure is compared to the inequality and challenges the Rainbow Nation continues to experience.

Advertisement

The question is: how can New Zealand’s All Blacks withstand and overcome the hitherto unshakable wave of momentum and spirit the Springboks play with under
Erasmus in this weekend’s World Cup final?

It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and
leaders that will call time on their international careers at week’s end.

The backlash and toxicity that followed their historic home series loss to Ireland was a humbling and public reminder of the expectation of being an All Black.

It’s why Saturday’s World Cup final is personal for the men wearing black.

All Blacks social media managers were forced to delete a thousand comments an hour following the humbling series loss in Wellington.
</item 3>

<item 4>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
It’s been far from smooth, having overseen their historic defeats to Argentina as well as the 2-1 series loss to Ireland on home soil.

A famous 35-23 win over the Springboks at Ellis Park saw him save his career, with Scott Robertson being lined up on the side.

But it wasn’t enough for New Zealand Rugby to back him before this year’s Test season got underway, with Robertson announced as Foster’s successor.

“Am I at peace?” Foster repeated a question during his World Cup team announcement when asked about where his mental state was ahead of his last Test in charge
of the All Blacks.

“Yeah, I’m reasonably peaceful. I love the word but I’m not sure I’m peaceful right now.”

Advertisement

He added: “I’ve just started to get my back nice and straight from last year and now you’re trying to make me hunched over again!

“It’s the life of this business, it’s a tough game.
</item 4>

<item 5>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
The All Blacks are 18-point favourites for the opening semi-final.

It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

“The past sets us up beautifully for both teams,” Foster said.

“We do know each other but we don’t know each other terms of a Rugby World Cup, so at this stage in the tournament it’s new territory for us and as we’ve seen
World Cups are very different.”

“We have learnt to greatly respect Argentina. They have a rich history of overachieving at the World Cup and have done a fantastic job to get here at the same
level we are, so it is going to be a heck of a game.”

Advertisement

Foster said the All Blacks, who suffered a shock semi-final loss to Eddie Jones’ England in 2019, never consider themselves favourites.

“Firstly, you’ve never heard us say we’re favourites,” he said.
</item 5>

<item 6>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 6>

<item 7>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The pain of the days and weeks after that dark July, as well as the bumps and bruises along the way, including as recently as August when they suffered their
biggest defeat to the Boks to lose the inaugural and prestigious Qatar Cup, still runs deep.

“It was quite special to hear some of the things those certain players said from the heart. You know, you think you know someone but once they open up and
really be vulnerable in front of a team, it was quite special to hear,” back-rower Dalton Papali’i said this week.



“It was truly something special when you hear those boys open up. You know those leaders have been through a lot in their careers, the highs and the lows, the
off-field and on-field stuff, and for them to open up, it sort of makes you feel welcomed and you want to play this game for them.
</item 7>

<item 8>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The All Blacks will experience a changing of the guard next season.

Indeed, the All Blacks’ social media channels described the team selection for the World Cup final as the “Last Dance”.

2015 World Cup winner Dane Coles will call time on his career, while his tight-five teammates Brodie Retallick and Sam Whitelock, who is bidding to become the
first player to win the Webb Ellis Cup for a third time, will head overseas.

Fellow World Cup winners Aaron Smith and Beauden Barrett will also head to Japan.

Coles, who was labelled by Samisoni Taukei’aho as a “Grandad” type figure in the All Blacks, has even been denied the chance of a farewell send-off after
missing selection.

Advertisement

“Yeah, tough,” said coach Ian Foster of the conversation with the 36-year-old around leaving him out.
</item 8>

<item 9>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Brodie Retallick will come off the bench.

Elsewhere, Mark Telea has been recalled ahead of Leicester Fainga’anuku. The tryscoring machine was axed for the quarter-final after missing a team curfew.

“That’s the team we think best for this week,” Foster said.

“Mark has done his time. He made a mistake and accepted what was happening but you don’t linger in that space. He’s been our form winger. We have got a faith in
him and think he is in good place to play this game.

Advertisement

“We really enjoyed Leicester Fainga’anuku]’s game last week and he should be proud of that and its’s a tough selection. It’s a chance for us to get Mark on the
park.”

The All Blacks are 18-point favourites for the opening semi-final.
</item 9>

<item 10>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We don’t want to talk too much about post-game stuff, but that was a tough one, probably the toughest I have had as a coach. He took it the way I expected him
to. He’s a champion.”

Papali’i said the All Blacks were intent on sending their leaders out on a high.

“There’s a few leaders in the team, it’s going to be their last game in the jersey, so it’s a big game for us and even the other players that are staying,
knowing that we want to send those boys out on a high,” he said.

“How much they put into the jersey. We are standing on their…what’s that quote? It’s standing on the shoulders of giants.”

The generational change isn’t just occurring on the field but off it, too.

Advertisement

Foster will depart the All Blacks set up after 12 years, the last four as head coach.
</item 10>
=== END ITEM id=8ca928b0738990f5 ===

=== ITEM id=38888692befb3f60 ===
Gold sentences and facts:
Sentence 0: In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.
  Fact 0.0: The Wallabies had a World Cup flop.
  Fact 0.1: The Queensland Rugby Union reaffirmed its position.
  Fact 0.2: The Queensland Rugby Union is committed to moving to a centralised high-performance model.
  Fact 0.3: The Queensland Rugby Union will not follow the NSW Waratahs by handing over the keys to Rugby Australia.
Sentence 1: He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.
  Fact 1.0: A person will be tasked with integrating the programs.
  Fact 1.1: RA is trying to centralise all of the Super Rugby's high-performance programs.

Retrieved items:
<item 1>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
His appointment reflects RA’s desire to go down an administrative path rather than former coach of any standing like Millard would have offered.

Horne, who joined World Rugby in 2015 as a general manager of high-performance before moving into his current role, will sit at the top of the high-performance
tree, with the next Wallabies coach to report into him.

Advertisement

During his tenure with World Rugby, the Sydneysider has played an important role in setting up competitions, high-performance structures and pathways across
developing countries.

He also played a role in helping Super Rugby integrate Fiji Drua and Moana Pasifika into the competition.

Those successes played out in the recent Rugby World Cup, with Chili, Fiji, Georgia and Portugal delivering encouraging performances that positively reflected
World Rugby’s desire to spread and build the game. Horne briefly worked at Saracens as a general manger.
</item 1>

<item 2>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU also expressed its satisfaction that RA was willing to discuss the annual $1.7m being reinstated, which had been withheld because of the Covid pandemic
to keep the game afloat.

Meanwhile, The Roar understands the ACT Brumbies have sent RA confirmation of solvency for the next 12 months.

RA was forced to take out a $40m loan from Ares to keep the game afloat.

Advertisement

While RA has around another $15m left to spend from the loan, the governing body is looking to borrow up to another $60m so they’re in the strongest possible
position to capitalise on the two major money-making events – the 2025 British and Irish Lions series and home men’s World Cup in 2017 – and succeed.
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 5>

<item 6>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=38888692befb3f60 ===
