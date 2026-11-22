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

=== ITEM id=b2265c976eba00a3 ===
Gold sentences and facts:
Sentence 0: It gave the JackJumpers a game-high 19-point lead and just about put the result to bed after Sydney had lost Jaylin Galloway (shoulder) and Jonah Bolden (fouled out).
  Fact 0.0: It gave the JackJumpers a game-high 19-point lead.
  Fact 0.1: The lead just about put the result to bed.
  Fact 0.2: Sydney had lost Jaylin Galloway (shoulder).
  Fact 0.3: Sydney had lost Jonah Bolden, who fouled out.
  Fact 0.4: The lead came after Sydney had lost Jaylin Galloway and Jonah Bolden.
Sentence 1: 9:20 p.m. — Nathaniel Lowe makes the play to retire Carroll, and that will bring the Rangers back up to bat with a 10-0 lead.
  Fact 1.0: At 9:20 p.m., Nathaniel Lowe makes the play to retire Carroll.
  Fact 1.1: The play by Nathaniel Lowe will bring the Rangers back up to bat.
  Fact 1.2: The Rangers will be back up to bat with a 10-0 lead.

Retrieved items:
<item 1>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
15 Rangers at Astros 8:15 p.m. Fox TEX 2, HOU 0 (TEX, 1-0) Oct. 16 Rangers at Astros 4:37 p.m. Fox or F TEX 5, HOU 4 (TEX, 2-0) Oct. 18 Astros at Rangers 8:03
p.m. FS1 HOU 8, TEX 5 (TEX, 2-1) Oct. 19 Astros at Rangers 8:03 p.m. FS1 HOU 10, TEX 3 (TIE, 2-2) Oct. 20 Astros at Rangers* 7:07 p.m. FS1 HOU 5, TEX 4 (HOU,
3-2) Oct. 22 Rangers at Astros* 8:03 p.m. FS1 TEX 9, HOU 2 (TIED, 3-3) Oct.
</item 1>

<item 2>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
😱



📺: FOX pic.twitter.com/83qi1EGjO2 — FOX Sports: MLB (@MLBONFOX) November 1, 2023

Rangers 7, Diamondbacks 0

9:08 p.m. 2-RUN DOUBLE — Who needs Adolis Garcia? Jankowski comes through with a massive 2-run double and puts the Rangers up 7-0. Texas is all over the
Diamondbacks' bullpen.

The Rangers tack on two more runs 😳



📺: FOX pic.twitter.com/AWYICKRMiq — FOX Sports: MLB (@MLBONFOX) November 1, 2023

9:06 p.m. — Luis Frias strikes out Taveras and has a chance to wiggle out of this against Jankowski.

9:03 p.m. — Disaster. Christian Walker had a chance to turn two but couldn't hold onto the ball.
</item 2>

<item 3>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
Even though they out-hit the Rangers, Arizona's bats hardly put any pressure on Texas' pitching.

MORE: Watch 2023 World Series games live with Fubo (free trial)

If you missed any of the action from Game 3 where the Rangers took a 2-1 series lead over the Diamondbacks, The Sporting News had you covered with live updates
and highlights below.

Rangers vs. Diamondbacks final score

1 2 3 4 5 6 7 8 9 F Rangers 0 0 3 0 0 0 0 0 0 3 Diamondbacks 0 0 0 0 0 0 0 1 0 1

World Series result, highlights from Game 3

(All times are Eastern.)

Ninth inning: Rangers 3, Diamondbacks 1

10:56 p.m.
</item 3>

<item 4>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
— Semien strikes out to end the inning for the Rangers.

10:33 p.m. — The game was briefly paused while the public address announcer asked fans to stop throwing paper airplanes onto the field. That's where this game
is right now.

Rangers 11, Diamondbacks 1

10:29 p.m. HOME RUN — For the first time since the third inning, the Rangers have a run. Jonah Heim blasts a home run to right field to extend Texas' lead back
to 10.

The Heim Hammer 💥



📺: FOX pic.twitter.com/G4pO4pz9jv — FOX Sports: MLB (@MLBONFOX) November 1, 2023

10:24 p.m. — Jankowski collides with Semien in foul territory but makes a nice catch for the final out of a quick inning for Bradford.

10:20 p.m. — Cody Bradford is into the game to pitch for Texas.
</item 4>

<item 5>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The Rangers found themselves one strike away from a championship twice in 2011. They couldn't finish the job. On Wednesday night, they'll have a chance to
avenge that painful finish.

Texas found out before Tuesday's Game 4 that slugger Adolis Garcia's season was over, but a talented lineup didn't miss a beat. The Rangers erupted for five
runs in the second inning and five more in the third, bludgeoning Arizona's bullpen to take a 10-0 lead in the blink of an eye.

Corey Seager came through with his third home run of the series, while Marcus Semien drove in five of the Rangers' first 10 runs — including a three-run blast
that put Texas up by double digits.

FAGAN: Three things Diamondbacks need to flip script on World Series

The Diamondbacks have earned a reputation for not backing down, and to their credit, they didn't.
</item 5>

<item 6>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Christian Walker had a chance to turn two but couldn't hold onto the ball. The Rangers have the bases loaded with one out.

8:59 p.m. — Lowe keeps the line moving and laces a single up the middle. The Rangers have two runners on with one out. This one could get even more out of hand
for Arizona. Kyle Nelson's night is done.

8:58 p.m. — Jung hits a one-out single for his second hit of the night.

8:53 p.m. — Thomas flies out to left to end the inning.

8:51 p.m. — Gurriel hits a two-out single to give the Diamondbacks a pulse. That'll bring up Alek Thomas.

8:49 p.m. — Heaney notches two quick outs to start the bottom of the second.

8:44 p.m. — The inning ends on a nice throw by Emmanuel Rivera, but what an inning it was for Texas with five runs.
</item 6>

<item 7>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The @Rangers came out swinging!



📺: FOX pic.twitter.com/Je43lpXXMi — FOX Sports: MLB (@MLBONFOX) November 1, 2023

8:37 p.m. — Taveras walks, and Travis Jankowski quickly singles up the middle to give the Rangers runners on first and second with two outs. Semien is up with a
chance to extend Texas' lead.

Rangers 1, Diamondbacks 0

8:35 p.m. WILD PITCH — Jung scores on a ball that gets away from Moreno! Castro's pitch in the dirt took a bad bounce, and there was no play at the plate. The
Rangers lead early.

The @Rangers strike first in Game 4!



📺: FOX pic.twitter.com/E5FCruQby7 — FOX Sports: MLB (@MLBONFOX) November 1, 2023

8:33 p.m.
</item 7>

<item 8>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Gun import Jordon Crawford has exploded late to lead the Tasmania JackJumpers to a club-record score in a 105-95 NBL road win over the defending champion Sydney
Kings.

The American guard opened the final period at Qudos Bank Arena on Sunday with four consecutive three-pointers and scored all of his team’s points on a stunning
14-2 run.

It gave the JackJumpers a game-high 19-point lead and just about put the result to bed after Sydney had lost Jaylin Galloway (shoulder) and Jonah Bolden (fouled
out).

Facebook Twitter Whatsapp Reddit Email Share

The Kings never gave up but could not close the gap as Crawford finished with a game-high 32 points.

Fellow import Milton Doyle and Majok Deng each scored 17 for Tasmania, who recorded their second win over Sydney in as many attempts this season.

“I had a couple of opportunities where I’m open and I shot the ball,” Crawford said.
</item 8>

<item 9>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers, who trailed by seven at three-quarter time, kept alive their record of winning every fourth term this season, but it wasn’t enough.

Brisbane captain Sobey led from the front as defensive specialist Mitch Norton kept Tasmania’s import Crawford (16 points) relatively quiet until the final
period.

Tyrrell Harrison and Josh Bannan (10 points each) were important for the Bullets, while Chris Smith (11) and Isaac White (10) gave solid contributions off the
bench.

The gut-wrenching loss ended a horror round five for the JackJumpers, who lost at home to Melbourne United before going down to the Bullets.

Advertisement

It left them with a 4-4 record after a strong 4-2 start to the season.
</item 9>

<item 10>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
7:35 p.m. — For all of the Rangers' losses, a win tonight with Nathan Eovaldi and Jordan Montgomery in line to start the next two games would put Texas in a
terrific position.

7:18 p.m. — Max Scherzer is also off the World Series roster for Texas. He's been replaced by Brock Burke.

7:07 p.m. — It's being reported that Adolis Garcia is off the World Series roster. His season is over. Ezequiel Duran has been added to the roster.

6:45 p.m. — Look who spent some time together before Game 4? The Diamondbacks are earning Chris Russo's trust.

6:25 p.m. — Bruce Bochy said he's anticipating making a roster decision on Adolis Garcia tonight. Garcia has a moderate oblique strain.

5:10 p.m. — Adolis Garcia is not in the lineup for Texas.
</item 10>
=== END ITEM id=b2265c976eba00a3 ===

=== ITEM id=4cb1420388759b01 ===
Gold sentences and facts:
Sentence 0: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 0.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 0.1: Prabhakar Raghavan revealed in court testimony that Google paid $26.3 billion in 2021.
  Fact 0.2: The $26.3 billion was paid to multiple browsers, phones and platforms.
  Fact 0.3: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 0.4: The Verge reports this.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: The case (the sentence's subject, referred to as 'It') will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 4>

<item 5>
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
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 7>

<item 8>
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
</item 8>

<item 9>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 10>
=== END ITEM id=4cb1420388759b01 ===

=== ITEM id=58ec1a385b076a0d ===
Gold sentences and facts:
Sentence 0: Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense, especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.
  Fact 0.0: Todd Boehly is American.
  Fact 0.1: Todd Boehly is a co-owner of Chelsea.
  Fact 0.2: Expanding Chelsea's profile and reach in the U.S. would make sense under Todd Boehly's co-ownership.
  Fact 0.3: Catarina Macario is a USWNT international.
  Fact 0.4: Mia Fishel is a USWNT international.
  Fact 0.5: Catarina Macario plays her club football in the U.S.
  Fact 0.6: Mia Fishel plays her club football in the U.S.
  Fact 0.7: CBS Sports holds WSL rights.
Sentence 1: It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again.
  Fact 1.0: The date was 24 September.
  Fact 1.1: Chelsea suffered a home Premier League defeat on 24 September.
  Fact 1.2: The 24 September defeat was Chelsea's second home Premier League defeat in a row.
  Fact 1.3: Chelsea's home Premier League defeat before the 24 September defeat was against Nottingham Forest.
  Fact 1.4: Chelsea were booed off again after the 24 September defeat.
Sentence 2: Late checks will be made on the fitness of Reece James, Axel Disasi and Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.
  Fact 2.0: Late fitness checks will be made on Reece James.
  Fact 2.1: Late fitness checks will be made on Axel Disasi.
  Fact 2.2: Late fitness checks will be made on Nicolas Jackson.
  Fact 2.3: Benoit Badiashile could return from a long-term injury.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 1>

<item 2>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 3>

<item 4>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Arsenal represent an acid test of the progress and the start of a daunting run in the league; after them, Chelsea face Brentford, Tottenham, Manchester City,
Newcastle, Brighton and Manchester United. Pochettino, though, pushed back against the notion that he was seeking a first statement win at the club, something
to ignite his tenure. All he wants are points to drive his team up from 11th in the table.

Pochettino has appeared mired in a selection crisis since his arrival at Chelsea; James only one element of it and he will be without seven others through
injury against Arsenal.

“The moment we have all the squad at a high level and fit, we can say: ‘OK, we are going to Arsenal, Manchester City and we are going to beat you.’ At the
moment, with the circumstances, I don’t think this game is going to be a statement to say: ‘Be careful, Chelsea is coming,’ or something like this.”
</item 4>

<item 5>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 6>

<item 7>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Chelsea were given a penalty and Tottenham went down to 10 men. #TOTCHE pic.twitter.com/sOzTrxgBzm — NBC Sports Soccer (@NBCSportsSoccer) November 6, 2023

Watch in UK:

"AMID FRENZY AND FURY!" 🤯



Cole Palmer levels for Chelsea as Cristian Romero sees red! 🟥 pic.twitter.com/1oULwk1uOP — Sky Sports Premier League (@SkySportsPL) November 6, 2023

Nicolas Jackson goal disallowed for offside

Chelsea would see a third goal of the half disallowed as Nicolas Jackson put the ball in the net on 37 minutes but Raheem Sterling was offside in the buildup
just past the midfield stripe. There's very little argument to be had here, and the game played on.

While it was a decision that featured little, if any, controversy, it added to the mayhem of the half and contributed to the game's overall narrative.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 10>
=== END ITEM id=58ec1a385b076a0d ===

=== ITEM id=ee836eefa3d68925 ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP), and the Philippine Association of Colleges and Universities (PACU).
  Fact 0.3: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the company's board to five directors.

Retrieved items:
<item 1>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.
</item 1>

<item 2>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.
</item 3>

<item 4>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 4>

<item 5>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 5>

<item 6>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
</item 6>

<item 7>
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
</item 7>

<item 8>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 8>

<item 9>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.
</item 9>

<item 10>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
The $440 million price also reflects a +51% premium on the value of the 29 catalogs implied by Hipgnosis Songs Fund’s public market cap for the 30 days ending
September 13.

What happened today?

Amongst the news confirmed by HSF today (September 28) is that the firm’s Chair of its board since 2018, Andrew Sutch, 73, is stepping down, both as HSF board
Chair and as a director of the company.

HSF says it’s begun a process to recruit a new Chair; Sutch (pictured inset) will leave his post as and when this person is hired, or in any case before HSF’s
AGM next year (in Q4 2024).

In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five
directors.
</item 10>
=== END ITEM id=ee836eefa3d68925 ===

=== ITEM id=e29f34f9198bfec7 ===
Gold sentences and facts:
Sentence 0: You can find value in player and team props by gathering information from multiple sources and tracking how players and their teammates perform.
  Fact 0.0: Value can be found in player and team props.
  Fact 0.1: Value in player and team props can be found by gathering information from multiple sources.
  Fact 0.2: Value in player and team props can be found by tracking how players and their teammates perform.
Sentence 1: These can be team or individual player prop bets, such as which team is winning at halftime or the total number of receptions a player collects.
  Fact 1.0: Props can be team prop bets.
  Fact 1.1: Props can be individual player prop bets.
  Fact 1.2: A prop bet can be on which team is winning at halftime.
  Fact 1.3: A prop bet can be on the total number of receptions a player collects.

Retrieved items:
<item 1>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
They vary from player to team bets and lean heavily on the statistical side of the game. Some NBA prop examples include:

How many points a player will score

Number of assists or three-pointers a player will have

Which player will score the first basket of a game

You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team
will score in the game.

See more: NBA prop bets

NFL prop bets

The National Football League is the king of sports betting, and NFL betting sites have no qualms about showcasing every kind of possible bet type for it.

Bettors will find many prop bet opportunities, especially for the Super Bowl. During the regular season, one of the most popular prop bet options includes the
“anytime touchdown scorer” bet, where you can wager on a player to score a TD at any point in the game. Other prop bets can be focused on the team or even both
teams.
</item 1>

<item 2>
College Bowl Odds, Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season (Sporting News, 2023-12-15T20:37:00+00:00)
Consider them side bets made with the sportsbook about events or statistical milestones that happen in the game.

These can be team or individual player prop bets, such as which team is winning at halftime or the total number of receptions a player collects. Consider them
side bets made with the sportsbook about events or statistical milestones that happen in the game. Bowl game parlays: Parlays are multi-leg bets on a single bet
slip. All parts of the parlay must be successful for the bet to win. These offer larger payouts, but the chances of winning compared with those of an individual
bet are greatly reduced.

How to bet on college football bowl games

Betting on NCAA football bowl games has never been easier, thanks to online betting sites sprouting up across the country.

Create an account with any of the best sports betting apps to start betting. It’s a quick process. Below, we outline the steps for opening your own account.
</item 2>

<item 3>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Player props could focus on how many points a player scores or how many rebounds he pulls down. These are typically provided in an over/under format.

Then, team props can range from which team will score the first bucket to the total points scored in a quarter to which team will lead at halftime. The
selection is often robust, especially when high-profile teams play.

Futures bets are exactly what their name implies: bets on events that will happen in the future. Common futures markets include:

and much more. You can usually bet on NBA futures odds throughout the year, including well before the season starts and throughout the regular season. The
further away from the event, the more likely you will find heavier odds in your favor.

Is NBA betting legal in the U.S.?

In the US, the landscape of sports betting, specifically NBA betting, has evolved considerably. While NBA betting is permitted, its legality varies across the
nation. The significant turning point came in 2018 when the US Supreme Court made a pivotal decision.
</item 3>

<item 4>
College Bowl Odds, Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season (Sporting News, 2023-12-15T20:37:00+00:00)
In point spread betting, you’re wagering on the game with a spread factored in. The favorite must win by the point spread or more, while the underdog needs to
either lose by fewer points than the spread or win the game outright. Bowl game totals: The totals bet, commonly known as the over/under, requires you to
determine whether the combined score of both teams will end up over or under a predicted total. With this bet type, you’re not concerned with which team wins or
loses the game.

The totals bet, commonly known as the over/under, requires you to determine whether the combined score of both teams will end up over or under a predicted
total. With this bet type, you’re not concerned with which team wins or loses the game. Bowl game props: These can be team or individual player prop bets, such
as which team is winning at halftime or the total number of receptions a player collects.
</item 4>

<item 5>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
For the most part, you won’t come across too many exotic prop bets when using US sportsbooks, as many are just not legal bets. However, the largest sporting
events, such as the Super Bowl or March Madness, will find sportsbooks offering up a wider range of prop bets. That’s usually when you’ll find more exotic
props.

Sportsbook rules for prop bets

Before placing any kind of bet, we highly recommend you read the sportsbook’s terms and conditions. There, you will find all the rules for the various bets you
will make.

While most sportsbooks’ rules are similar, there are always small variations in how they handle certain situations that could impact your bets.

One rule that is typically used by most sportsbooks involves player props. If you bet on a player and they don’t play in the game, most sportsbooks void the bet
and refund your wager. But if that player is involved in the game even for a single play, then the bet stands.
</item 5>

<item 6>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
You’ll often come across NBA team prop bets that are based on a team’s points, such as the first team to reach 20 points, the first team to score in a game, or
whether the total points by both teams in a contest will end up being odd or even.

Team props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven by
individual performance.

These individual props are focused on statistical milestones during a game. The most popular bets typically focus on points, rebounds, assists, and three-point
field goals. You can also bet on combination wagers, such as points and rebounds together.

In most cases, you can bet on traditional statistics for the five starters on each team.

See more:

Tips for betting on NBA props

Every bettor should develop their strategies, but we do have three tips you can use for NBA props.
</item 6>

<item 7>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 7>

<item 8>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Let’s say you’re keeping an eye on the New York Jets vs. the Buffalo Bills, and you see the total shift due to an incoming winter storm. While many bettors will
scramble to get their bet in on the ‘under’ before the lines move too much, it’s a good idea to scope out the prop bets for individual players. The main lines
are shifting, but most of the best sports betting sites don’t have the personnel to keep tabs on every possible player, so you could find value in an ‘under’ on
a quarterback’s passing yards because the lines haven’t caught up yet. There’s greater variety: The three main bet types—moneylines, point spreads, and
totals—are constant in every major sport, but they don’t offer a lot in the way of variety. Prop bets do. They can add interest to a game that may seem
one-sided in the major markets. Larger sporting events, especially the Super Bowl, offer a massive variety of bets, ranging from the typical to the exotic and
everything in between.
</item 8>

<item 9>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
The NBA is one of the most popular sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.

Proposition bets, or props, are easy to understand, numerous, and cover both player props and team props to give you plenty of variety.

Below, we look at betting NBA props at the best NBA betting sites and share some key information to make informed bets.

What are NBA props?

A prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA, prop bets deal with both players and teams.
</item 9>

<item 10>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Use most of your budget on more stable bet types, such as moneylines, spreads, or totals. Focus only a small percentage of your bankroll on props.

We wouldn’t recommend using prop bets as your main bet type when building your strategy. Use most of your budget on more stable bet types, such as moneylines,
spreads, or totals. Focus only a small percentage of your bankroll on props. Exotic props are a guessing game: When large events like the Super Bowl roll
around, you’ll be inundated with exotic and entertaining prop bets. These are almost impossible to research, and pure guessing isn’t good for anyone’s bankroll
or betting strategy.

Prop bet examples for major sports

Every major sport with odds at the best sports betting apps will have prop bets. Below, we give examples of prop bets you might find when betting on major
American sports.

NBA prop bets

The NBA offers a wide range of prop bets. They vary from player to team bets and lean heavily on the statistical side of the game.
</item 10>
=== END ITEM id=e29f34f9198bfec7 ===

=== ITEM id=8ddca9147694683c ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison was the CEO of Alameda.
  Fact 0.2: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: Bankman-Fried had 'secret access' to customer money.
  Fact 0.4: The customer money included both cash and crypto.
  Fact 0.5: The government claimed this.
Sentence 1: He was concerned because it "seemed like a lot of money" from FTX customers was at risk.
  Fact 1.0: The person was concerned about the risk.
  Fact 1.1: The person was concerned because the money seemed like a lot of money.
  Fact 1.2: The money was from FTX customers.
  Fact 1.3: The money from FTX customers was at risk.

Retrieved items:
<item 1>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
In June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in customer money held in an internal database tracking the cash wired to
an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and Bankman-Fried spoke about it at the pickleball court at the resort in Nassau,
Bahamas. He asked his boss if things were OK. He was concerned because it "seemed like a lot of money" from FTX customers was at risk. "Sam said, we were
bulletproof last year. We aren't bulletproof this year," Yedidia testified. Yedidia said he asked when they would be bulletproof again. Bankman-Fried said he
wasn't sure, but it may be six months to three years. Yedidia said Bankman-Fried appeared "worried or nervous," which he said was atypical.
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
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 8>

<item 9>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 9>

<item 10>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 10>
=== END ITEM id=8ddca9147694683c ===
