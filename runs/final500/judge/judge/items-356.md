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

=== ITEM id=145e2930c47f1408 ===
Gold sentences and facts:
Sentence 0: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 0.0: This sentence follows the heading 'Love Story'.
  Fact 0.1: This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday.
  Fact 0.2: Swift was cheering on No.
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has said she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 5>

<item 6>
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
</item 6>

<item 7>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
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
</item 8>

<item 9>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 10>
=== END ITEM id=145e2930c47f1408 ===

=== ITEM id=c4f683bcf79c172e ===
Gold sentences and facts:
Sentence 0: More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
  Fact 0.0: Sam is one of the most brilliant and impactful people the industry has ever had.
  Fact 0.1: Sam is more important as one of the most generous and caring people the speaker knows than as one of the most brilliant and impactful people the industry has ever had.
  Fact 0.2: Sam is one of the most generous and caring people the speaker knows.
Sentence 1: While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s.
  Fact 1.0: Altman is not trained as an AI engineer.
  Fact 1.1: Altman is now 38.
  Fact 1.2: Altman has been seen as a Silicon Valley wunderkind since his early 20s.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board’s language.
  Fact 2.1: The prevailing theory floating around right now is based on the way these giant tech companies work.
  Fact 2.2: The described theory is the prevailing theory floating around right now.

Retrieved items:
<item 1>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 1>

<item 2>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
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
</item 3>

<item 4>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 4>

<item 5>
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
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 6>

<item 7>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 7>

<item 8>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 8>

<item 9>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 9>

<item 10>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 10>
=== END ITEM id=c4f683bcf79c172e ===

=== ITEM id=387637423bbe559d ===
Gold sentences and facts:
Sentence 0: This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google’s on the road to victory.
  Fact 0.0: This whole line of argument is called 'market definition'.
Sentence 1: Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.
  Fact 1.0: If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google is on the road to victory.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: Monday's trial hearing revealed the $26.3 billion Google spent in 2021 to make itself the default search engine across platforms.
  Fact 2.1: Monday's trial hearing revealed that Google tried to take its default search engine position further.
  Fact 2.2: Monday's trial hearing revealed that Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 2>

<item 3>
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
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 10>
=== END ITEM id=387637423bbe559d ===

=== ITEM id=a34f21a94c8e2e25 ===
Gold sentences and facts:
Sentence 0: It has also decided to make Beeper Mini free for the time being, given how “chaotic” things have been over the past few days.
  Fact 0.0: The party referred to as 'it' has decided to make Beeper Mini free for the time being.
  Fact 0.1: Things have been "chaotic" over the past few days.
  Fact 0.2: The decision to make Beeper Mini free was made given how "chaotic" things have been over the past few days.
Sentence 1: In addition, Beeper Mini will be free for the time being given the unstable nature of Beeper’s status and Apple’s attempts at shutting it down.
  Fact 1.0: Beeper Mini will be free for the time being.
  Fact 1.1: Beeper's status is unstable.
  Fact 1.2: Apple has attempted to shut Beeper Mini down.
  Fact 1.3: Beeper Mini is free because of the unstable nature of Beeper's status and Apple's attempts at shutting it down.
Sentence 2: Beeper Mini was made possible because the team behind the app has managed to reverse-engineer the iMessage protocol, they say.
  Fact 2.0: Beeper Mini was made possible because the team behind the app reverse-engineered the iMessage protocol.
  Fact 2.1: The team behind the app says it has managed to reverse-engineer the iMessage protocol.

Retrieved items:
<item 1>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
It’s $1.99/mo, not $2.99/mo; 12/6/23, 2:33 PM ET added “[sent it]” to clarify that the Beeper app does not look like an iPhone app.
</item 1>

<item 2>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
That won’t necessarily prevent Apple from sending Beeper legal a Cease and Desist, of course. Apple previously sued spyware maker NSO Group to block it from
using Apple’s services, and it could likely make a legal case here, as well, if it chose to. What may hold it at bay is the Digital Markets App (DMA), a law in
Europe that says big tech companies will have to have an interoperable interface for their chat networks. There are also stirrings of antitrust efforts in the
U.S., where Apple is under federal scrutiny, which could make for bad timing to target Beeper, too.

But to be fully trusted, Beeper Mini will need to be audited by a third party — something it has not yet done. In addition, Beeper uses certificate pinning,
which makes network traffic analysis more difficult to perform in order to verify its claims. The company says its external audit is still “in progress” but it
has performed an internal audit.
</item 2>

<item 3>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
The company is publishing those results on its blog along with a detailed, more technical description of how Beeper Mini works.

For example, the team explains here how it needed to build a new service, called Beeper Push Notification service (BPNs), to make the service work:

A persistent connection to APNs is needed to be notified of new incoming messages in real-time. On an iPhone, an APNs connection is maintained by the operating
system, and connected at all times. In Beeper Mini, the connection can only be maintained when the app is running, since Android does not support APNs natively.

To work around this limitation, the team built BPNs to connect to Apple’s servers on the user’s behalf when the app isn’t running.

In tests, Beeper Mini worked as described, able to send iMessage texts from an older Android phone on Google Fi’s wireless network with its own phone number
(not associated with an Apple ID) to an iPhone 15 Pro Max with a different phone number that is associated with an Apple ID.
</item 3>

<item 4>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
The push to bring iMessage to Android users today adds a new contender. A startup called Beeper, which had been working on a multi-platform messaging
aggregator, is now launching a new app called Beeper Mini which will allow Android users to send and receive end-to-end encrypted iMessage chats for just $1.99
per month. Beeper Mini was made possible because the team behind the app has managed to reverse-engineer the iMessage protocol, they say.

“That’s the big breakthrough,” explains Beeper co-founder and CEO Eric Migicovsky, previously the founder of smartwatch startup Pebble. “We’re not actually a
middleman anymore. The research that we’ve done is actually reverse-engineering the iMessage protocol, down to the lowest layer of the protocol. So Beeper Mini
doesn’t use a Mac server as a relay like all the other apps — they have a Mac Mini in a data center somewhere.
</item 4>

<item 5>
Beeper Mini is back in operation after Apple’s attempt to shut it down (TechCrunch, 2023-12-11T18:08:48+00:00)
Senator Elizabeth Warren (D-Mass.) shamed Apple for its intervention in a post on X, formerly Twitter, saying “Green bubble texts are less secure. So why would
Apple block a new app allowing Android users to chat with iPhone users on iMessage? Big Tech executives are protecting profits by squashing competitors.
Chatting between different platforms should be easy and secure,” she wrote.

On Monday, Beeper announced on its blog that it has updated the Beeper Mini app to fix the issue that “caused messages not to be sent or received.” It hasn’t
said exactly how it’s made that happen, but notes that the security of Beeper Mini remains unchanged — promising it still includes local, end-to-end encryption
as previously described. Migicovsky told TechCrunch he can’t share anything on that topic beyond what’s already in the blog post.

Though Beeper Mini is running again, there is a caveat: phone number registration isn’t yet working. That means users will have to sign in with an Apple ID.
</item 5>

<item 6>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
That means that Beeper Mini can function as a true iMessage client, supporting high-resolution photos and videos, threads, replies, read receipts, direct
messages and group chats, tapback emoji reactions, editing and unsending messages, as well as support for stickers, GIFs, voice notes and more. Not yet
supported are features like live location sharing, message effects and support for FaceTime audio and video calls. Users will not need an Apple ID to use Beeper
Mini.

“It’s a full-blown iMessage app,” Migicovsky says. “For all intents and purposes, Beeper Mini looks like an iPhone [sent it].”

Apple, then, may not be able to simply block Beeper Mini texts automatically if it wanted to take action against the company, though the co-founder admits how
Apple will react is still uncertain.

However, he points to a provision in copyright law, The Digital Millennium Copyright Act (DMCA 1201 F), which says that reverse-engineering for the purposes of
interoperability is protected.
</item 6>

<item 7>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
And when you send a message, you’re actually sending a message to the Mac Mini, which then forwards it to iMessage,” he explains. “Beeper Mini is a native
implementation of the iMessage protocol.”

Beeper does not have access to the contents of users’ messages, the company claims. And unlike the recently paused efforts by Sunbird, which had been trying to
solve the same problem, messages are not sent in clear text.

Instead, the message you send from an Android phone using Beeper Mini is end-to-end encrypted to the recipient, the startup says. It’s encrypted on the device
before it leaves the app. Encryption keys are exclusively stored on your phone within the Android filesystem, similar to other apps like Signal and WhatsApp.
The app doesn’t connect to any servers at Beeper itself, only to Apple servers, the way a “real” iMessage text would.
</item 7>

<item 8>
Beeper Mini is back in operation after Apple’s attempt to shut it down (TechCrunch, 2023-12-11T18:08:48+00:00)
Beeper Mini, the app bringing blue bubble iMessage texts to Android users, is back in operation on Monday after a long weekend that saw Apple putting an end to
Beeper’s services, claiming it was a security risk. Now, the startup behind the new app says it has resumed functioning, but hasn’t disclosed how that’s been
made possible. In addition, Beeper Mini will be free for the time being given the unstable nature of Beeper’s status and Apple’s attempts at shutting it down.

Founded by former Pebble smartwatch founder Eric Migicovsky, Beeper had originally been working on an app that aggregated all of a user’s chat apps into a
single interface. However, upon the discovery of new technology that allowed it to reverse-engineer the iMessage protocol, it set out to build a new app called
Beeper Mini that solely focused on bringing iMessage chats to Android users for $1.99 per month. That included support for high-quality photos and videos,
tapback reactions, typing indicators, read receipts and more.
</item 8>

<item 9>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
Initially, Beeper launched a multi-network chat app that supported around 15 different networks, including WhatsApp, Messenger, Telegram, Signal, Slack,
Instagram, LinkedIn, Discord, Google Chat, Android SMS, iMessage and others.

“Until I started Beeper, I didn’t really understand this whole iMessage thing. I didn’t really understand why people loved it so much,” Migicovsky says. “I
think one of the reasons is that it’s so deeply built into iOS, that it’s second nature for you as an iPhone user. When you want to get in touch with someone,
you open the Messages app.”

His experience with messaging had differed, though, having lived in Europe and Asia at times, where he collected friends across a wider network of applications.
He didn’t understand the draw that iMessage had in the U.S. in particular, as he was simply left out of friends’ iMessage chats for being the friend with a
green bubble.
</item 9>

<item 10>
Beeper Mini is back in operation after Apple’s attempt to shut it down (TechCrunch, 2023-12-11T18:08:48+00:00)
The app quickly climbed into the top 20 in the Play Store’s charts and saw over 100,000 downloads in its first 48 hours, the company claims.

Unfortunately, Beeper Mini’s success was short-lived. Only days after launching, Apple on Friday night took action against the startup, having found a way to
stop Beeper Mini’s messages from being passed from Android phones to its servers. The Cupertino tech giant explained that it needed to shut down the app because
it “posed significant risks to user security and privacy, including the potential for metadata exposure and enabling unwanted messages, spam, and phishing
attacks.” Essentially, Apple said that it couldn’t verify that Beeper’s claims of end-to-end encryption were true, and therefore it needed to take “steps to
protect our users by blocking techniques that exploit fake credentials in order to gain access to iMessage.”

As the Beeper team scrambled for a fix, U.S. Senator Elizabeth Warren (D-Mass.)
</item 10>
=== END ITEM id=a34f21a94c8e2e25 ===

=== ITEM id=c2bdd27b66efad60 ===
Gold sentences and facts:
Sentence 0: They want more than three points, but it's early enough that they absolutely could settle for a field goal if it got to that point.
  Fact 0.0: The team in question wants more than three points.
  Fact 0.1: It is still early enough in the game.
  Fact 0.2: The team in question absolutely could settle for a field goal if the game got to that point.
Sentence 1: 11:03 p.m. — The Vikings are using their two timeouts, but there isn't much they'll be able to do once this hits fourth down.
  Fact 1.0: At 11:03 p.m., the Vikings are using their two timeouts.
  Fact 1.1: There isn't much the Vikings will be able to do once the situation reaches fourth down.
Sentence 2: The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running game for Kevin O'Connell and the defense overall under Brian Flores.
  Fact 2.0: Josh Dobbs is a new player on the Vikings.
  Fact 2.1: Josh Dobbs keeps producing the same offensive results the Vikings enjoyed with Kirk Cousins.
  Fact 2.2: Josh Dobbs's results have come with more help from the running game under Kevin O'Connell.
  Fact 2.3: Josh Dobbs's results have come with more help from the overall defense under Brian Flores.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
“If before the game someone offered Philly to put its defense against Zach Wilson on a need-field-goal drive starting inside his own 15 with no timeouts and a
minute left, you think they would have taken it?”

In the situation Sunday, the Jets’ offense had netted 19 yards on its previous three possessions, producing one field goal on a drive that gained 16 of those
yards. The Eagles had already suffered three turnovers. They knew Wilson would be taking over deep in his territory with very little time remaining.

Advertisement

“To me, these are exactly the type of factors that a coach needs to consider when deciding,” the exec said. “I think they were ignored here.”

The situation Philly faced against the Jets was the seventh time since 2000 that a team leading by 1-2 points in the final two minutes faced third down with
7-10 yards to go from the minus-40 yard line to midfield, with the opponent holding no timeouts.
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 3>

<item 4>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 4>

<item 5>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Three of the seven teams passed, resulting in one first down, one sack and Hurts’ interception. Of these three teams, only the Eagles lost the game.

3. The 49ers lost Sunday when their rookie kicker missed from 41 yards. Is San Francisco wise trusting a championship roster to a rookie’s foot come playoff
time?

The 49ers raised eyebrows among some when they used their second-highest pick in the 2023 draft — a third-round choice, 99th overall — for a kicker. The
selection of Michigan’s Jake Moody gave the 49ers a cost-effective successor to veteran Robbie Gould. General manager John Lynch called Moody a potentially
“foundational” player whose stock in the draft was so high, other teams were trying to trade ahead of San Francisco for a shot at him.

Who do you want lining up the potential winning kick in a Super Bowl?
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 9>

<item 10>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 10>
=== END ITEM id=c2bdd27b66efad60 ===

=== ITEM id=33559b84338cdb4a ===
Gold sentences and facts:
Sentence 0: Halloween isn’t the only holiday on the horizon now that October is drawing to a close.
  Fact 0.0: Halloween is not the only holiday on the horizon.
  Fact 0.1: October is drawing to a close.
Sentence 1: If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
  Fact 1.0: The reader may be in a new relationship this holiday season.
  Fact 1.1: The reader may be faced with the opportunity to introduce their significant other to their family this holiday season.

Retrieved items:
<item 1>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
However, it’s likely you’ve found yourself mulling over an array of reasons why the holidays may not be the right time for such an introduction.

The holiday season is often perceived as the most wonderful time of year, but that’s not necessarily the case for everyone.
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 2>

<item 3>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Halloween isn’t the only holiday on the horizon now that October is drawing to a close. Once the clock strikes midnight on October 31st, Mariah Carey will
defrost, peppermint mochas will inevitably return, and the 2023 holiday season will have officially begun.

Thankfully for those looking to get in the festive spirit, Govee’s Matter-ready LED Strip Light M1 is on sale for an all-time low of $44.99 ($15 off) at Amazon
when you clip the on-page coupon. You can also buy it for $3 more direct from Govee through October 31st, which is still one of its better prices to date.

While handy throughout the year, Govee’s 6.56-foot (2m) light strip is particularly convenient during Halloween and other holidays. Not only can the bright,
tunable LEDs flash and blink to the beat of your music, but they can also act as a festive piece of decor that adds a pop of color to your holiday parties.
</item 3>

<item 4>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.
</item 4>

<item 5>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 5>

<item 6>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
So when fellow video extraordinaire Vjeran Pavic (The Verge’s supervising producer) recently surprised me with a very cute birthday note and this magical little
gadget, I was elated.

And if that wasn’t enough, it has a carabiner hook. It is everything I have ever wanted in a rubber case and more. To have both my SD card reader and all my
cards in one place is priceless (well, actually $39.95). Thank you, Vjeran. <3

A mobile tripod for holiday snaps

Jess Weatherbed, news writer

When you go on holiday with your partner or family, you generally get stuck with three options when it comes to taking group photographs: a cramped selfie,
leaving someone out to take the picture, or asking a total stranger to take it for you. Not to be dramatic or anything, but I’d rather not ruin my vacation by
trying to figure out which individuals nearby are the least likely to dip the minute I hand over my phone.
</item 6>

<item 7>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
What’s particularly great about this strip, though, is that it supports the newer Matter protocol, which allows you to control the lights with a wide array of
smart home platforms.

Govee LED Strip Light M1 Matter $ 45 $ 60 25 % off $ 45 $ 45 $ 60 25 % off The customizable Govee M1 Matter Compatible LED Strip Light is 6.56 feet long and can
display millions of colors and shades of white. You can also use it to create light patterns that sync with music. $45 at Amazon$48 at Govee

Though they launched last year, Google’s flagship wireless earbuds continue to impress us with their terrific sound quality and good noise cancellation, which
is why they’re one of our favorite pairs of wireless earbuds. We’re also fans because of all the other perks they offer, like multipoint connectivity, long
battery life, and a good transparency mode.
</item 7>

<item 8>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.
</item 8>

<item 9>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I’d recommend pulling your partner aside and saying: ‘How are we doing?’ Like a temperature check,” she added. “Those are points where the two of you can come
and remind each other: ‘We are a team in this.’ Or, the two of you could go on a walk and separate from the family, just for a little bit of time to reset
within the relationship.”

During the holiday season, your partner may be seeing how you interact with your family for the first time. But that’s not necessarily a bad thing, according to
the relationship experts. “I think introducing a significant other to family is a really vulnerable act, and that vulnerability is going to be what allows for
greater intimacy,” Lurie said. “When your significant other sees you with your family, they may see another side of you that they may not have had access to
before.”

(Getty Images)

On the other hand, your partner could learn that your relationship with your parents and extended family members is complex.
</item 9>

<item 10>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Blink Video Doorbell (with Blink Outdoor 4) $ 90 $ 180 50 % off $ 90 $ 90 $ 180 50 % off Blink’s Video Doorbell is the best if you’re looking for a
budget-friendly buzzer that offers motion-activated recording and alerts, night vision, two-way audio, and up to two years of battery life. $90 at Amazon

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Some more deals to kickstart your weekend
</item 10>
=== END ITEM id=33559b84338cdb4a ===
