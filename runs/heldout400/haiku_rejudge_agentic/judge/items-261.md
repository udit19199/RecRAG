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

=== ITEM id=98d48e1a406fff80 ===
Gold sentences and facts:
Sentence 0: The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the Springboks’ blitz defence in the wet conditions.
  Fact 0.0: The All Blacks had their chances.
  Fact 0.1: The All Blacks' usual safe handling was a feature throughout the World Cup.
  Fact 0.2: The All Blacks' usual safe handling evaded them against the Springboks' blitz defence in the wet conditions.
  Fact 0.3: The All Blacks lost their first-up World Cup match to France.
Sentence 1: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 1.0: The situation described comes despite Argentina beating the All Blacks last year in Christchurch.
  Fact 1.1: The situation described comes despite a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnold, Mat Ryan, Andrew
Redmayne, South Americans, the tournament, Asian confederation, Frank Farina, Qatar, FIFA, The Roar
</item 1>

<item 2>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Advertisement

Robertson will have an almighty task on his hand, with multiple generational players including Dane Coles, Sam Whitelock, Brodie Retallick and Aaron Smith to
call time on their international careers. Others like Ardie Savea, Mo’unga, Beauden Barrett and Shannon Frizell will also head to Japan.

While Foster didn’t want to take anything away from the Springboks’ win, he said the game had some issues to sort out regarding the officiating.

“I’m proud of our guys, to go down to that red card so early and fight our way back and give ourselves a chance is pretty special,” the departing All Blacks
coach said.

“I don’t think it went wrong in any clear [place on the pitch], it was a real arm wrestle, both teams have their moment, incredibly proud of the way we fought
and to get within a whisker of pulling it off is heart-breaking.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Ardie Savea -PARTICIPATED_IN-> World Cup quarter-final
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Ardie Savea
- Rassie Erasmus -WORKS_FOR-> Springboks

Entities: Ardie Savea, Springboks, Beauden Barrett, THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic
final, Robertson, Sam Whitelock, Brodie Retallick, Shannon Frizell, officiating, RWC final, Mo’unga, Dane Coles, Foster, Japan, All Blacks
</item 2>

<item 3>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Minutes later and Kolisi was off, as he cannoned into Savea. He would later return, much to the anger of the New Zealanders in the crowd who were calling for
justice.

Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide. It came back to haunt them, as their
usually silky and composed skill sets failed them.

That much was true when the TMO intervened after Smith thought he had a try in his final Test, but an error at the lineout cancelled out the five-pointer.

Advertisement

Eventually the All Blacks did have a try, as Mark Telea, the All Blacks’ best and biggest attacking threat all night, managed to get on the outside and draw in
two defenders before offloading to Beauden Barrett. But Mo’unga missed the conversion to leave them trailing by a point.

Graph facts:
- Beauden Barrett -PARTICIPATED_IN-> World Cup final
- Beauden Barrett -PARTICIPATED_IN-> Test match
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Beauden Barrett
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Beauden Barrett
- Savea -PARTICIPATED_IN-> World Cup final

Entities: Boks, Beauden Barrett, Savea, THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final, RWC
final, Mo’unga, Kolisi, Smith, Mark Telea, All Blacks, The Roar | Sports Writers Blog
</item 3>

<item 4>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
STADE DE FRANCE – Twenty-eight years between drinks and drama was always on the cards between these two great foes. Little did we know just how much.

But just like that famous win at Ellis Park where Nelson Mandela handed over the William Webb Ellis Cup to Francois Pienaar, it was the Springboks who once
again prevailed 12-11 in an instant classic.

Unlike the blue skies that greeted Johannesburg and the Boeing 747 plane that sent shivers down the spines of everyone in attendance in 1995, rain and dampness
poured down on the Stade de France.

Facebook Twitter Whatsapp Reddit Email Share

Yet, nothing was going to dampen this evening as 80,065 fans poured in.

In a match of controversy, where cards of yellow and red were handed out quicker than a game of Uno, thundering shots both legal and illegal were put on by
rugby’s two greatest sides.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> William Webb Ellis Cup
- Subtle as a sledgehammer, a pop icon returns -MENTIONS-> Nelson Mandela
- Rassie Erasmus -WORKS_FOR-> Springboks

Entities: 1995 win at Ellis Park, Nelson Mandela, Francois Pienaar, William Webb Ellis Cup, Boeing 747, Springboks, THE GREATEST! Boks beat All Blacks in
instant classic to claim record FOURTH RWC in controversial, dramatic final, RWC final, Stade de France, Ellis Park, Johannesburg, All Blacks, The Roar
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

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Christchurch, Opening semi-final,
Argentina-All Blacks match last year, Argentina-All Blacks match, All Blacks-England semi-final, England, Foster, Sydney, Rugby World Cup, Eddie Jones, All
Blacks, Argentina, New Zealand, The Roar
</item 5>

<item 6>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
“This group is very proud to be All Blacks, very proud to play for their country and have worked incredibly hard to get here.

“While I am so proud of what we did and how we worked we have to give it to South Africa, they are a quality team. They have had a few close victories in this
tournament and are a different class. It’s special for them but it equally hurts for us. I can’t ask for much more than we did, we couldn’t put more out on the
park.”

Advertisement

Controversy was on display from the outset, as Frizell was shown a yellow card for an illegal cleanout on Springboks hooker Bongi Mbonambi. The incident saw the
hooker, who was cleared earlier in the week by World Rugby following an alleged racial slur on England back-rower Tom Curry, replaced by Deon Fourie in the
third minute.

The penalty allowed Pollard to step up and bang over three points.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -PUBLISHED_BY-> The Roar
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> 1995 win at Ellis Park
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Nelson Mandela
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Francois Pienaar
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> William Webb Ellis Cup

Entities: THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final, Frizell, Bongi Mbonambi, Tom Curry,
Deon Fourie, illegal cleanout, racial slur, RWC final, South Africa, England, Pollard, World Rugby, All Blacks, The Roar | Sports Writers Blog
</item 6>

<item 7>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.

Graph facts:
- Arnold -PARTICIPATED_IN-> playoff against Peru
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Australia
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -PUBLISHED_BY-> The Roar | Sports Writers Blog
- Arnold -WORKS_FOR-> Socceroos
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Peru

Entities: ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Graham Arnold, Socceroos, Socceroos World Cup
qualification campaign, Asian playoff against United Arab Emirates, playoff against Peru, Peru, World Cup, Football Australia, World Cup finals, Qatar,
Argentina, The Roar, United Arab Emirates
</item 7>

<item 8>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
A missed Richie Mo’unga conversion following Beauden Barrett’s 58th minute try – the only five-pointer of the match – ultimately proved the difference, as
Handre Pollard’s four three-pointers led the Springboks to World Cup glory.

The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the
Springboks’ blitz defence in the wet conditions.

All night the All Blacks’ timing was off and the forced ball from Savea out wide just as the men in black saw space and looked to open up the Springboks in the
80th minute proved to be killer blow.

Wayne Barnes, who was booed relentlessly, called a scrum reset as the seconds ticked by.

Advertisement

Moments later, their hearts sunk as the Springboks managed to win the scrum and then hang onto the ball.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, Richie Mo’unga, Beauden Barrett, Handre Pollard, Savea, Wayne Barnes, World Cup final, THE GREATEST! Boks beat All Blacks in instant
classic to claim record FOURTH RWC in controversial, dramatic final, France, All Blacks, The Roar | Sports Writers Blog
</item 8>

<item 9>
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

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Los Pumas, South Africa, All
Blacks, New Zealand, The Roar
</item 9>

<item 10>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Argentina, 2015 final, game in
2020, Wallabies, rugby, All Blacks, New Zealand, The Roar
</item 10>
=== END ITEM id=98d48e1a406fff80 ===

=== ITEM id=7cf0958bf0f45760 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang testified during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Huang said Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 0.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 1.0: Judge Lewis Kaplan intervened to ask whether Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 1.1: Bankman-Fried admitted he had been told about that money only after Judge Lewis Kaplan intervened.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that Bankman-Fried, the FTX founder, acted in good faith.
  Fact 2.2: The defense argued that Bankman-Fried, the FTX founder, never meant to commit fraud or steal.
  Fact 2.3: The defense argued that Bankman-Fried, the FTX founder, basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.

Graph facts:
- Thane Rehn -PARTICIPATED_IN-> SBF Trial
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Thane Rehn
- Thane Rehn -SAID-> SBF
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Thane Rehn
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> stealing customer funds

Entities: Thane Rehn, Larry David, stealing customer funds, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand, Alameda Research, Fortune, FTX, Sam Bankman-Fried, Tom Brady, government
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sam Bankman-Fried fraud trial, Federal Court in New York City,
financial crimes, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, Sam Bankman-Fried, Manhattan, Reuters,
CNBC
</item 2>

<item 3>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Easton
- Easton -PARTICIPATED_IN-> SBF Trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried

Entities: Easton, Bankman-Fried, In the end, the FTX trial was about the friends screwed along the way, Paper Bird, Effective Altruism movement, Can Sun,
Apollo, missing funds, FTX trial, FTX, The Verge
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Adam Yedidia
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission

Entities: The jury finally hears from Sam Bankman-Fried, Alameda Research, allow_negative, Ellison, Singh, Wang, Sam Bankman-Fried, The Verge
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
</item 5>

<item 6>
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

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Matt Huang
- Matt Huang -WORKS_FOR-> Sequoia Capital
- Today the FTX jury suffered through a code review -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Paradigm

Entities: Matt Huang, Paradigm, Alameda, Getty Images, money laundering, Bahamas, New York, US, FTX, Sex, Signal messages, and sabotaging FTX's code: SBF execs
and Bahamas roommates tell all in court, Yuki Iwamura, Sam Bankman-Fried, CNBC, fraud, Bloomberg
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, theft, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda Research, Rehn, Fortune, FTX, Sam Bankman-Fried, government
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Paradigm
- Meet Alliance DAO’s latest accelerator cohort of startups building in crypto winter -MENTIONS-> Paradigm
- Matt Huang -WORKS_FOR-> Paradigm
- Paradigm -PARTICIPATED_IN-> class-action lawsuit
- Paradigm -ACCUSED_OF-> defrauding customers

Entities: Paradigm, Bankman-Fried, Alameda, Signal, Bahamas, FTX, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in
court, Huang, Dawn Giel, board of directors, CNBC
</item 8>

<item 9>
Graph query row:
category: relationship
entity: Sam Bankman-Fried
relationship: MENTIONS -> PUBLISHED_BY -> PUBLISHED_BY -> MENTIONS
related: Sam Bankman-Fried -> SBF’s trial starts soon, but how did he — and FTX — get here? -> TechCrunch -> SBF Trial: The latest updates from the FTX
collapse’s courtroom drama -> Paradigm
page: None
passage: None
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 10>
=== END ITEM id=7cf0958bf0f45760 ===

=== ITEM id=cb04fee30c63e674 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified.
  Fact 0.1: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.2: Bankman-Fried said it was okay to continue withdrawing money from the account as long as the amount was less than FTX's total trading revenue.
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried was a former billionaire.
  Fact 1.3: Bankman-Fried got his start in finance at Jane Street.
  Fact 1.4: Bankman-Fried got his start in finance after MIT.
  Fact 1.5: Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.6: Bankman-Fried persuaded Yedidia to later join him as a developer at FTX.
Sentence 2: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 2.0: The speaker said it was Bankman-Fried's decision.
  Fact 2.1: The speaker said Bankman-Fried was the CEO of FTX.
  Fact 2.2: The speaker said it was Bankman-Fried's decision because Bankman-Fried was the CEO of FTX.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 3.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 3.2: The defense countered that the FTX founder acted in good faith.
  Fact 3.3: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 3.4: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Graph query row:
person: Sam Bankman-Fried
connected_entity: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
relationship: MENTIONS
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
article_mention: Josh Naftalis
mention_relationship: MENTIONS
</item 1>

<item 2>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Easton
- Easton -PARTICIPATED_IN-> SBF Trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried

Entities: Easton, Bankman-Fried, In the end, the FTX trial was about the friends screwed along the way, Paper Bird, Effective Altruism movement, Can Sun,
Apollo, missing funds, FTX trial, FTX, The Verge
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda Research, FTX, Nishad Singh, Wang, Sam Bankman-Fried, The Verge
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Adam Yedidia
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission

Entities: The jury finally hears from Sam Bankman-Fried, Alameda Research, allow_negative, Ellison, Singh, Wang, Sam Bankman-Fried, The Verge
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.

Graph facts:
- Michael Lewis -PARTICIPATED_IN-> Michael Lewis interview
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Michael Lewis
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Going Infinite
- Michael Lewis -SAID-> FTX
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Going Infinite, Michael Lewis, Alameda, Sam Bankman-Fried was a terrible boyfriend, Conspiracy to commit fraud, Risk-neutral, Gambling addict, Staff
revolt within Alameda Research over lost millions and general financial chaos, money laundering, Ellison, Sam Bankman-Fried, fraud, The Verge
</item 5>

<item 6>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda

Entities: Bankman-Fried, Alameda Research, In the end, the FTX trial was about the friends screwed along the way, Bill Clinton, Tony Blair, Silicon Valley
startup culture, allow_negative, FTX trial, FTX, Wang, Yedidia, Silicon Valley, The Verge
</item 6>

<item 7>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> SBF Trial
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google’s hardware event
- Founders, are events useful? -MENTIONS-> Pixel 8
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Pixel 8
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Pixel 8 Pro

Entities: teens’ startup, ad-free tier, Pixel 8, Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier, Jacquelyn,
Southern District of New York courthouse, Bankman-Fried, SBF and FTX trial, TechCrunch, TechCrunch+, the defense, FTX, Google, Rebecca, the prosecution
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Gary Wang, Caroline Ellison, Alameda Research, defense, Nishad Singh, Signal, FTX trial, The FTX trial is bigger than Sam
Bankman-Fried, LaVigne, witness tampering, FTX, Sam Bankman-Fried, government, Twitter
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, In the end, the FTX trial was about the friends screwed along the way, Guarding Against Pandemics, Gabe Bankman-Fried, Singh, K5,
Modulo, Paper Bird, FTX Foundation, FTX trial, FTX, Anthropic, The Verge
</item 9>

<item 10>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Caroline Ellison, Bankman-Fried, Alameda Research, wire fraud, Nishad Singh, embezzlement, money laundering, conspiracy, political
donations, In the end, the FTX trial was about the friends screwed along the way, Joe Bankman, FTX trial, FTX, Yedidia
</item 10>
=== END ITEM id=cb04fee30c63e674 ===
