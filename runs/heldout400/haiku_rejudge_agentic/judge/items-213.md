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

=== ITEM id=54b15c6462271e64 ===
Gold sentences and facts:
Sentence 0: “There was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic,” said an Anthropic spokesperson, Sally Aldous.
  Fact 0.0: Sally Aldous is an Anthropic spokesperson.
  Fact 0.1: Sally Aldous said there was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic.
Sentence 1: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 1.0: A year has passed since Sam Altman catapulted ChatGPT to global fame.
  Fact 1.1: Sam Altman has become Silicon Valley's sought-after voice on the promise of artificial intelligence.
  Fact 1.2: Sam Altman has become Silicon Valley's sought-after voice on the potential dangers of artificial intelligence.
  Fact 1.3: Sam Altman's exit from OpenAI was sudden and mostly unexplained.
  Fact 1.4: Sam Altman's sudden and mostly unexplained exit from OpenAI brought uncertainty to the industry's future.
Sentence 2: More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
  Fact 2.0: Sam is one of the most brilliant and impactful people the industry has ever had.
  Fact 2.1: The speaker says that being one of the most brilliant and impactful people the industry has ever had is less important than Sam being one of the most generous and caring people the speaker knows.
  Fact 2.2: Sam is one of the most generous and caring people the speaker knows.
Sentence 3: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 3.0: The board's language supports a prevailing theory.
  Fact 3.1: The way giant tech companies work supports a prevailing theory.
  Fact 3.2: The prevailing theory is floating around right now.
  Fact 3.3: The prevailing theory is based on the board's language and the way giant tech companies work.

Retrieved items:
<item 1>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, Satya Nadella, conference, How ego and fear fuelled the rise of artificial intelligence, Hoffman, Altman, Sun Valley, Kevin Scott, The deal,
LinkedIn, Microsoft, The Age, artificial intelligence, OpenAI, Tesla
</item 1>

<item 2>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Researchers inside OpenAI, Google and other companies thought this rapidly improving technology could be a path to AGI. But Amodei was unhappy about the
Microsoft deal because he thought it was taking OpenAI in a really commercial direction. He and other researchers went to the board to try to push Altman out,
according to five people familiar with the matter. After they failed, they left. Like DeepMind’s founders before them, they worried that their new corporate
overlords would favour commercial interests over safety.

In 2021, the group of about 15 engineers and scientists created a new lab called Anthropic. The plan was to build AI the way the effective altruists thought it
should done – with very tight controls. “There was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic,” said an Anthropic spokesperson,
Sally Aldous.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Musk
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Page
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Mark Zuckerberg
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Peter Thiel

Entities: How ego and fear fuelled the rise of artificial intelligence, effective altruists, Amodei, Microsoft deal, commercial interests, Sally Aldous, group
of about 15 engineers and scientists, creation of Anthropic, attempt to remove Sam Altman from OpenAI, DeepMind, Microsoft, The Age, Google, safety, Anthropic
</item 2>

<item 3>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, OpenAI, OpenAI's board, vote of no confidence, WTF is going on at OpenAI? We have theories, Sam Altman
</item 3>

<item 4>
Graph query row:
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
publisher: TechCrunch
published_on: 2023-10-07T20:15:26+00:00
</item 4>

<item 5>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
They believed that AI could find a cure for cancer or solve climate change, but they worried that AI bots might do things their creators had not intended. If
the machines became more intelligent than humans, the Rationalists argued, the machines could turn on their creators. Thiel had become enormously wealthy
through an early investment in Facebook and through his work with Musk in the early days of PayPal. He had developed a fascination with the singularity, a trope
of science fiction that describes the moment when intelligent technology can no longer be controlled by humanity.

With funding from Thiel, Yudkowsky had expanded his AI lab and created an annual conference on the singularity. Years before, one of Hassabis’ two colleagues
had met Yudkowsky, and he snagged them speaking spots at the conference, ensuring they’d be invited to Thiel’s party. Yudkowsky introduced Hassabis to Thiel.
Hassabis assumed that lots of people at the party would be trying to squeeze their host for money.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, Thiel, Rationalists, Yudkowsky, annual conference on the singularity, Thiel’s party, singularity, Hassabis, How ego and fear fuelled the rise of
artificial intelligence, AI lab, PayPal, The Age, Facebook, science fiction, cancer
</item 5>

<item 6>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, Page, Rosewood hotel, Thiel, How ego and fear fuelled the rise of artificial intelligence, Victoria Krakovna, Bloomberg Businessweek, Hoffman,
PayPal, ethics board, break away from Google, DeepMind, The Age, Google, Silicon Valley
</item 6>

<item 7>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Peter Thiel
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> Peter Thiel
- Peter Thiel -PARTICIPATED_IN-> President Trump's 2016 campaign
- Peter Thiel -COMPETES_WITH-> Satya Nadella
- Musk -COMPETES_WITH-> Peter Thiel

Entities: Peter Thiel, LinkedIn, TechCrunch, AI, Elon Musk, Joe Biden, Olaf Scholz, Y Combinator, OpenAI, Worldcoin, Sam Altman ousted as OpenAI’s CEO, Sam
Altman, Reid Hoffman, U.S. congressional hearings, Emmanuel Macron
</item 7>

<item 8>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.

Graph facts:
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Microsoft
- Microsoft -MENTIONS-> cloud gaming

Entities: Microsoft, AI, Fortune, Silicon Valley, OpenAI, Ilya Sutskever, OpenAI board, Quora, Adam D’Angelo, Tasha McCauley, Georgetown Center for Security and
Emerging Technology, YCombinator, OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board
did', Mira, Helen Toner
</item 8>

<item 9>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
AI was once laughed off. Now the richest men in Silicon Valley were shelling out billions to keep from being left behind. The lost ethics board

When Musk invested in DeepMind, he broke his own informal rule – that he would not invest in any company he didn’t run himself. The downsides of his decision
were already apparent when, only a month or so after his birthday spat with Page, he again found himself face to face with his former friend and fellow
billionaire. Larry Page, co-founder of Google. Credit: Bloomberg The occasion was the first meeting of DeepMind’s ethics board, on August 14, 2015. The board
had been set up at the insistence of the startup’s founders to ensure that their technology did no harm after the sale. The members convened in a conference
room just outside Musk’s office at SpaceX. But that’s where Musk’s control ended. When Google bought DeepMind, it bought the whole thing. Musk was out.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, SpaceX, How ego and fear fuelled the rise of artificial intelligence, Larry Page, DeepMind ethics board, first meeting of DeepMind’s ethics
board, birthday spat, DeepMind, The Age, Google, Silicon Valley, artificial intelligence, Bloomberg
</item 9>

<item 10>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> false information
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Altman
- How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement -MENTIONS-> Altman
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Altman
- WTF is going on at OpenAI? We have theories -MENTIONS-> Altman

Entities: false information, Altman, consumer protection laws, AI, Fortune, San Francisco, OpenAI, CEO summit, Asia-Pacific Economic Cooperation conference,
U.S. Federal Trade Commission, investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false information
through its chatbot, public data, chatbot, OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the
board did'
</item 10>
=== END ITEM id=54b15c6462271e64 ===

=== ITEM id=1d090f423659a286 ===
Gold sentences and facts:
Sentence 0: As for the All Blacks, they looked lost at times during this World Cup cycle and it’s still faintly remarkable that head coach Ian Foster survived last summer as the home defeats to Ireland, South Africa and Argentina piled up.
  Fact 0.0: The All Blacks looked lost at times during this World Cup cycle.
  Fact 0.1: It is still faintly remarkable that head coach Ian Foster survived last summer.
  Fact 0.2: Ian Foster survived last summer as the home defeats piled up.
  Fact 0.3: The home defeats were to Ireland.
  Fact 0.4: The home defeats were to South Africa.
  Fact 0.5: The home defeats were to Argentina.
Sentence 1: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 1.0: Argentina knocked over the All Blacks last year in Christchurch.
  Fact 1.1: The event in this sentence came despite Argentina knocking over the All Blacks last year in Christchurch.
  Fact 1.2: There was a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
New Zealand played some scintillating rugby in victory (AFP via Getty Images)

They led 13-0 and though they may not have realised, no team had ever squandered a lead that large in a World Cup quarter-final. Not that they would have
expected an easy ride from there.

They conceded a penalty straight from the restart as their kick chase was illegally impeded and Sexton duly slotted the three.

Ireland had made something of a statement during the Haka as their fans drowned it out with a rousing rendition of Fields of Athenry and the players formed a
figure of eight in memory of ex-international and Munster coach Anthony Foley, who tragically died in 2016, as they stared down their opponents.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sexton
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> AFP
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> AFP
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> AFP
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> AFP

Entities: Sexton, AFP, Getty Images, The Independent, Ireland, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues, New
Zealand, Haka, Fields of Athenry, Anthony Foley, Munster, World Cup quarter-final
</item 1>

<item 2>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
And then there were those 37 phases… the most painful finish in Irish rugby history.

Ireland were left bereft after an agonising defeat (PA Wire)

Instead of being feted as the heroes to finally overcome that insurmountable obstacle, the Ireland players will still have to answer endless questions about the
heavy burden of previous knockout failures and fans will endure at least four more years of choking jokes at their expense.

As for the All Blacks, they looked lost at times during this World Cup cycle and it’s still faintly remarkable that head coach Ian Foster survived last summer
as the home defeats to Ireland, South Africa and Argentina piled up.

Yet somehow, he hung on and turned things round. He has the unwanted title of being the first New Zealand coach to lose a World Cup pool match but now a
semi-final awaits next Friday evening against an Argentina side that, frankly, are a tier below them. Forget the hand-wringing ahead of the tournament, the All
Blacks are one very small step away from another World Cup final.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, South Africa, World Cup semi-final, All Blacks, The Independent, Argentina, Ireland, Irish rugby, Ian Foster, World Cup pool match,
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues
</item 2>

<item 3>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
PARIS – The All Blacks have downplayed Rieko Ioane’s gestures towards Ireland fans and retiring great Johnny Sexton, as the three-time world champions said they
have learned to “respect Argentina” ahead of their semi-final on Saturday (6am AEDT).

It comes as Michael Cheika embraced the underdog status and said he was hoping Los Pumas “inspired” Argentina “not by our results but by our ambition”, as the
former Wallabies coach readies for another clash against the All Blacks and a second semi-final.

Facebook Twitter Whatsapp Reddit Email Share

Ioane copped a spray from Sexton following their 28-24 nail-biting win over Ireland.

It came after the lethal back cupped his ear at Irish supporters at the Stade de France after the final whistle, as well as putting his finger to his lips in a
‘shush gesture’.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Johnny Sexton
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea

Entities: Johnny Sexton, Michael Cheika, semi-final, 28-24 win over Ireland, ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on
‘favourites’ New Zealand, Paris, Stade de France, Wallabies, All Blacks, Argentina, Ireland, Rieko Ioane, The Roar
</item 3>

<item 4>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
But the first real in-match statement came on 27 minutes as New Zealand-born Aki – perhaps the player of the World Cup so far who had made a tournament-high 61
carries, 33 dominant carries and 23 defenders beaten heading into the weekend – brilliantly jinked inside two defenders, fended off another and powered through
a gap to dive over the line.

Yet more All-Black class saw Ardie Savea go over in the corner after quick ball was shipped wide but with scrum-half Aaron Smith in the sin-bin for a deliberate
knock-on, Ireland narrowed the gap to 18-17 just before the break as another New Zealand-born Irish star, Jamison Gibson-Park, sniped from a lineout maul and
somehow stretched out for the try.

The All Blacks extended an incredible record of leading at half-time of their quarter-final in every single World Cup but that lead had never felt more fragile.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> All Blacks
- Ardie Savea -PARTICIPATED_IN-> RWC final
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Ardie Savea
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> All Blacks
- Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal -MENTIONS-> All Blacks

Entities: Ardie Savea, All Blacks, The Independent, Ireland, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues, Aki, Aaron
Smith, Jamison Gibson-Park, New Zealand, World Cup quarter-final
</item 4>

<item 5>
Graph query row:
article: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand
published_on: 2023-10-18T20:55:55+00:00
team: Argentina-All Blacks match last year
opponent: 28-24 win over Ireland
match_location: Christchurch
passage: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
PARIS – The All Blacks have downplayed Rieko Ioane’s gestures towards Ireland fans and retiring great Johnny Sexton, as the three-time world champions said they
have learned to “respect Argentina” ahead of their semi-final on Saturday (6am AEDT).

It comes as Michael Cheika embraced the underdog status and said he was hoping Los Pumas “inspired” Argentina “not by our results but by our ambition”, as the
former Wallabies coach readies for another clash against the All Blacks and a second semi-final.

Facebook Twitter Whatsapp Reddit Email Share

Ioane copped a spray from Sexton following their 28-24 nail-biting win over Ireland.

It came after the lethal back cupped his ear at Irish supporters at the Stade de France after the final whistle, as well as putting his finger to his lips in a
‘shush gesture’.
</item 5>

<item 6>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
While Ireland will rue the 37 phases that ended in heartbreak, they may just go down in New Zealand rugby history. Tired bodies made tackle after tackle and not
a single penalty was given away.

The delight at the final whistle showed this wasn’t just any quarter-final victory and while Ireland ponder just what they have to do to finally win a World Cup
knockout match, the All Blacks can dream of lifting the Webb Ellis Cup for a fourth time. The pain and ecstasy of sport writ large.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, All Blacks, The Independent, Ireland, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues, New Zealand,
Webb Ellis Cup, quarter-final victory
</item 6>

<item 7>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
Sign up to our free sport newsletter for all the latest news on everything from cycling to boxing Sign up to our free sport email for all the latest news Please
enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read
our privacy notice Thanks for signing up to the

Sport email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

This time was supposed to be different. This Ireland team were different… are different. But, come the end, the result was the same. On quarter-final weekend,
for the eighth time in 10 Rugby World Cups, Ireland head home defeated and the wait for a first-ever knockout win goes on.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -PUBLISHED_BY-> The Independent
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -PUBLISHED_BY-> The Independent
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -PUBLISHED_BY-> The Independent
- Taylor Swift is Time’s Person of the Year -PUBLISHED_BY-> The Independent
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -PUBLISHED_BY-> The Independent

Entities: The Independent, Rugby World Cups, Ireland, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues
</item 7>

<item 8>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
Foster will leave his role after the World Cup, to be replaced by all-conquering Crusaders boss Scott Robertson, but he can still exit with the greatest prize
of all. And frankly, who would bet against the All Blacks now?

The opening-night nerves against France seem a long time ago and they ultimately had too much for Ireland in this simply incredible, back-and-forth Test match
that may genuinely go down as one of the best of all time.

After early penalties from Richie Mo’unga and Jordie Barrett, they scored a scintillating try through Leicester Fainga’anuku, who was only in the starting XV
due to Mark Telea’s breach of team protocol earlier in the week. Beauden Barrett expertly dinked over the top, collected his own chip and when the ball was
spread wide, Fainga’anuku and Rieko Ioane neatly combined to send the former over in the corner.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, Richie Mo’unga, Beauden Barrett, Scott Robertson, Jordie Barrett, Mark Telea, Leicester Fainga’anuku, Foster, France, All Blacks, The
Independent, Ireland, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues, Crusaders, Test match
</item 8>

<item 9>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
This is the greatest Ireland team of all time, they may still even be the best team in the world but on this day, the quarter-final curse, not to mention an
inspired All Blacks side, ruled once more.

This defeat will hurt more than any other. Truthfully, since 1991 when Michael Lynagh’s late try handed Australia victory, Ireland hadn’t come close to ending
their last-eight hoodoo. Never mind winning one, they haven’t led a World Cup quarter-final since the first half of the 1995 defeat to France, a mammoth 10,353
days ago.

But here they had a legitimate chance to reverse the curse. Multiple times they got within a point of the All Blacks, only for their opponents to agonisingly
pull away once more. And then there were those 37 phases… the most painful finish in Irish rugby history.

Graph facts:
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> Australia
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> Australia
- Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars -MENTIONS-> Australia
- IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series -MENTIONS-> Australia
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Australia

Entities: Australia, France, All Blacks, The Independent - Sports, Ireland, Michael Lynagh, 1991 Australia victory, 1995 defeat to France, quarter-final curse,
Irish rugby, Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues, World Cup quarter-final
</item 9>

<item 10>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
The All Blacks eventually prevailed in Paris (Reuters)

The second half followed a similar, rollercoaster pattern as every time Ireland got close, their opponents found a way to stretch the lead. Mo’unga’s sumptuous
dummy and sleight of hand from first-phase lineout ball froze two Irish defenders and led to a scything break before he fed speedster Will Jordan to race over
the whitewash.

That extended the gap to eight points but a penalty try as an Irish driving maul was illegally hauled down closed it to 25-24 with 17 minutes to go and sent
Codie Taylor for 10 minutes in the sin-bin.

Yet this All Blacks team suddenly have a resilience about them and brushed off the underdog status they had been given in the build-up to this game as their
tireless back row continually disrupted breakdowns and a disciplined defence refused to yield.

Jordie Barrett added a penalty to take the score to 28-24 and that’s how it eventually stayed.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Mo’unga
- Jordie Barrett -PARTICIPATED_IN-> RWC final
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Jordie Barrett
- Jordie Barrett -PARTICIPATED_IN-> Test match
- Mo’unga -PARTICIPATED_IN-> RWC final

Entities: Jordie Barrett, Mo’unga, Will Jordan, Paris, game, Reuters, All Blacks, The Independent, Ireland, Thirty-seven phases of agony define Ireland’s
greatest heartbreak as curse continues, Codie Taylor
</item 10>
=== END ITEM id=1d090f423659a286 ===

=== ITEM id=eae433065510f2dc ===
Gold sentences and facts:
Sentence 0: The artists’ groups noted in their statement on Tuesday that the European Commission “has on several occasions acknowledged concerns about the impact of the ruling and made clear its intention to find a balanced solution.
  Fact 0.0: The artists' groups noted in their statement on Tuesday that the European Commission has on several occasions acknowledged concerns about the impact of the ruling.
  Fact 0.1: The artists' groups noted in their statement on Tuesday that the European Commission made clear its intention to find a balanced solution.
Sentence 1: We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
  Fact 1.0: The speaker continues to work through the process with the European Commission.
  Fact 1.1: The speaker is focused on addressing the European Commission's questions at this stage.
  Fact 1.2: The speaker is focused on addressing any identified concerns at this stage.
Sentence 2: And some of these staff will also be deployed within the European Commission,” they added.
  Fact 2.0: The speaker added that some of these staff will also be deployed within the European Commission.
Sentence 3: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 3.0: Earlier actions by the entity referred to as 'its' were focused on concerns about the spread of illegal content.
  Fact 3.1: Earlier actions by the entity referred to as 'its' were focused on concerns about disinformation related to the Israel-Hamas war.

Retrieved items:
<item 1>
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

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -MENTIONS-> disinformation
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> disinformation

Entities: disinformation, illegal content, risk management, content moderation, TechCrunch, European Commission, formal DSA investigation on X, noyb, X’s
adtech, complaint against X’s adtech, dark patterns, advertising transparency, data access for researchers, formal request for information, Elon Musk’s X faces
first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design
</item 1>

<item 2>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
At the time the EU’s then competition chief, Margrethe Vestager, trumpeted the outcome as setting what she dubbed “new rules for how Amazon operates its
business in Europe” — arguing the settlement meant the ecommerce giant could “no longer abuse its dual role” as marketplace operator and own brand retailer; and
further claiming the outcome would ensure “competing independent retailers and carriers as well as consumers will benefit from these changes opening up new
opportunities and choice.”

Given the commitments the EU already extracted from Amazon after those earlier investigations it’s interesting to see the bloc remains concerned over levers the
ecommerce behemoth still has its disposal to potentially influence competitive outcomes on its marketplace vis-à-vis iRobot vs rival robot vacs.

Since then Amazon has also been named a gatekeeper under the EU’s shiny new ex ante competition regulation, the Digital Markets Act — which lists its
marketplace as a core platform service. This designation means Amazon must comply with a series of up-front obligations, which include restrictions on self
preferencing.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- Commission -SAID-> competition concerns
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: competition concerns, iRobot, EU, Amazon’s iRobot purchase sucks up formal competition concerns in EU, Margrethe Vestager, Amazon’s iRobot purchase,
Amazon marketplace, robot vacs, self preferencing, TechCrunch, settlement, Europe, Amazon, Digital Markets Act
</item 2>

<item 3>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.

Graph facts:
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> Washington
- La desigualdad del estrés por calor -MENTIONS-> Washington
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Washington
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Washington
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Washington

Entities: Washington, AI, Warner Music Group, copyright law, Universal Music Group, US Senate, Chuck Schumer, Music Business Worldwide, Artificial Intelligence
(AI) Insight Forum, Dream Track, creative copyright, intellectual property, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned
from Dennis Kooker’s speech about AI, Dennis Kooker, Sony Music Entertainment
</item 3>

<item 4>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> AI systems
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> AI systems
- Biden’s AI EO -MENTIONS-> AI systems
- Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment -MENTIONS-> AI systems
- Sam Altman ousted as OpenAI’s CEO -MENTIONS-> AI systems

Entities: AI systems, large language models, AI, Google, Universal Music Group, Anthropic, Music Business Worldwide, Dream Track, Sony Music has issued nearly
10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI, United States Copyright Office, copyright law and policy issues,
Stability AI, fair use, copyrighted material, Ed Newton-Rex
</item 4>

<item 5>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 5>

<item 6>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> hate speech
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Saturday’s attacks
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> terrorist and violent content
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> risk assessment
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Commission services

Entities: Investigation into X over alleged disinformation, complaint handling, risk assessment, After warning, the EU opens investigation into X over alleged
disinformation, disinformation, illegal content, EU, TechCrunch, X, Elon Musk, Commission, DSA, Twitter
</item 6>

<item 7>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
Today’s announcement by the EU, following several months of in-depth examination looking at how the transaction might impact competition, lays out formal
concerns for Amazon to respond to. The EU’s objections focus on foreclosure risks.

The EU says its concerns focus on whether the deal will give Amazon the ability and incentive to foreclose iRobot’s rivals by deploying strategies aimed at
preventing competitors from selling RVCs on Amazon’s marketplace and/or making it harder for them to do so — such as by delisting rival products; reducing their
visibility in both non-paid (i.e. organic) and paid results (i.e. ads) displayed on its marketplace; limiting their access to certain widgets it operates (such
as the “other products you may like” feature) or “certain commercially-attractive product labels” (e.g. ‘Amazon’s choice’ or ‘Works With Alexa’); and/or by
directly or indirectly raising the costs of iRobot’s rivals to advertise and sell their RVCs on its marketplace.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase
- iRobot -MENTIONS-> intense competition
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: RVCs, Amazon’s marketplace, Amazon’s choice, Works With Alexa, other products you may like, foreclosure risks, iRobot, EU, Amazon’s iRobot purchase
sucks up formal competition concerns in EU, TechCrunch, Alexa, competition, Amazon
</item 7>

<item 8>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> risk assessment
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> illegal content
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> illegal content
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> illegal content
- Meta says it’s prioritizing livestreaming checks during Israel-Hamas war -MENTIONS-> illegal content

Entities: risk assessment, illegal content, moderation, EU, TechCrunch, formal infringement proceedings, Hamas’ terrorist attacks against Israel, deceptive UX
design, Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design, X, Elon Musk, Hamas,
transparency, Commission, Israel
</item 8>

<item 9>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, content moderation, TechCrunch, deceptive UX design, Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation,
transparency and deceptive UX design, first DSA probe in EU, illegal content risks, X, EU regulators, the Commission, crowdsourced opinions, diverse information
risks, harm exposure, Elon Musk, EU lawmakers
</item 9>

<item 10>
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

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Investigation into X over alleged disinformation
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> complaint handling
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> risk assessment
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> terrorist and violent content
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> hate speech

Entities: After warning, the EU opens investigation into X over alleged disinformation, disinformation, illegal content, TechCrunch, X, Elon Musk, European
Union, Israel-Hamas war, Commission, Digital Services Act, Twitter
</item 10>
=== END ITEM id=eae433065510f2dc ===
