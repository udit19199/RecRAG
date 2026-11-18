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

=== ITEM id=10249954203c6ed2 ===
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
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 4>

<item 5>
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
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 6>

<item 7>
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
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 10>
=== END ITEM id=10249954203c6ed2 ===

=== ITEM id=302a138eb0cfe227 ===
Gold sentences and facts:
Sentence 0: He is helping to usher in a youth movement up front for Argentina, with Julian Alvarez and Alejandro Garnacho considered the future of the forward line.
  Fact 0.0: He is helping to usher in a youth movement up front for Argentina.
  Fact 0.1: Julian Alvarez is considered the future of the forward line.
  Fact 0.2: Alejandro Garnacho is considered the future of the forward line.
Sentence 1: 2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.
  Fact 1.0: The kickoff is 2 hours away.
  Fact 1.1: Argentine journalist Gaston Edul is well-connected.
  Fact 1.2: A report from Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.

Retrieved items:
<item 1>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
With the 2023 MLS season closing earlier than expected for Inter Miami, eliminated from playoff contention, Lionel Messi will have significant time off before
the new campaign begins in February 2024.

With every new season come preseason friendlies, as the players hope to ramp up their match fitness prior to the start of the coming campaign. Players need game
time to build their bodies to the point where they can play full 90-minute matches, especially when they are contesting multiple games a week.

Inter Miami's preseason will be watched with extra observation as Messi draws considerable attention wherever he goes, home or abroad. The Argentina star had
looked like being part of a tour of China, although that's no longer the case, while the prospect of facing a familiar side from home is a tantalising one.

The Sporting News has compiled all the known preseason matches either confirmed or reported prior to the start of Inter Miami's 2024 season.
</item 1>

<item 2>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Details on Lionel Messi's injury as he battles "fatigue" and a "scar tissue issue".

La idea es que Leo Messi vuelva a sumar minutos mañana cuando Inter Miami juegue contra Cincinnati. pic.twitter.com/ePNWHb4vSs — Gastón Edul (@gastonedul)
October 6, 2023

Inter Miami vs FC Cincinnati lineups & team news

There was a report that Lionel Messi could finally make his return from injury, and while it felt unlikely that Messi would start three days after missing out
entirely, he appears on the bench today and could be available for a second half appearance if required.

Jordi Alba remains sidelined with a hamstring injury, missing since his injury just minutes before Messi, and without a publicly known timeline for a return.
Also missing is striker Leonardo Campana who was absent from the midweek defeat to the Chicago Fire with a muscle injury, although Gerardo Martino described his
issue as "minor."
</item 2>

<item 3>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Inter Miami season tickets for 2024 season see unprecedented jump in price

(Also, you'll notice a photo of Lionel Messi's jersey hanging in his locker in that tweet below, a pretty good indication that he'll return tonight)

Scene is set for Fan Appreciation Night 🏟️#MIAvCIN | 7:30 PM ET pic.twitter.com/9I07uq9hGW — Inter Miami CF (@InterMiamiCF) October 7, 2023

2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for
this match in some capacity. Edul reported that "the idea is for Leo Messi to get minutes when Inter Miami plays FC Cincinnati".

That makes it sound like Messi will only be available off the bench, but it's better than nothing to have a game-changer like him available at all.
</item 3>

<item 4>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Inter Miami no va a viajar a China. La gira que incluía dos amistosos se cae por temas burocráticos/comerciales. pic.twitter.com/8I0jBxg4Zz — Gastón Edul
(@gastonedul) October 30, 2023

Instead, Inter Miami met NYCFC in a hastily arranged friendly at DRV PNK Stadium on November 10 to honor Lionel Messi's most recent Ballon d'Or. The friendly is
being dubbed the "Noche d'Or" or "Night of Gold" as a nod to his latest achievement.
</item 4>

<item 5>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Meanwhile, Fabrizio Romano also shot it down from a journalistic standpoint.

There isn't enough of a break for him to reasonably integrate with Barcelona, and at 36 years old he's sure to welcome the extended time off.

Tata Martino on rumors of Messi playing on loan with Barcelona this winter:



"Is he going for fun? Maybe he's going to visit Barcelona. Other than that, I don't know anything else about it." pic.twitter.com/qJwoJOwkC6 — Felipe Cárdenas
(@FelipeCar) October 8, 2023

2 hours to kickoff: Lionel Messi's club Inter Miami were eliminated from MLS playoff contention after their latest defeat, a 1-0 loss to FC Cincinnati last
weekend. Messi had missed the previous four matches but returned to the field for the final 30 minutes. He was powerless to turn the tide of the game, after
Inter Miami had missed so many gilt-edged chances in the first half.
</item 5>

<item 6>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
@MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF #Messi𓃵 #cf97 #MLS pic.twitter.com/hp8KHzkwNB — Michelle Kaufman (@kaufsports) October 4, 2023

2 hours to kickoff: Tata Martino has played his cards extremely close to his chest when it comes to the fitness of Lionel Messi. Each week we see how
tight-lipped the Inter Miami boss has been regarding Messi's injury.

Prior to tonight's match against the Chicago Fire, Martino was asked about the return of Jordi Alba, and he stated that there is a timeline in place for Alba's
return. When asked if there's a timeline for Messi's return, Martino refused to acknowledge if there is.

Tata Martino when I asked about Lionel Messi's return timeline:



"I understand your desire to know the exact timelines, but if I had the exact timelines I would share them.



"I just don't have them."
</item 6>

<item 7>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
A reminder, this match officially kicks off at 7:39 p.m. ET so there will be a slight delay from the advertised 7:30 p.m. time.

Inter Miami fan news: the 5 supporters’ groups known as La Familia threatened to stage a protest this game re: ticket price hike for 2024, but team offered to
meet w/them next week, so protest off at least until next game. @MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF — Michelle Kaufman (@kaufsports) October
7, 2023

30 mins to kickoff: Lionel Messi's availability for Inter Miami is a welcome sight for the club, as they take the field today with their backs against the wall.
Inter Miami must win tonight to ensure they remain alive in the playoff race. A draw or a loss, however, would see them vulnerable to outside results.
</item 7>

<item 8>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
The team from Argentina hold a special place in the heart of Messi and Martino. Not only are both men from the city of Rosario, where Newell's are based, but
Messi developed at the club's youth academy and Martino played more than 400 senior games for the club before becoming head coach in 2012.

"I am delighted to welcome my beloved Newell's to our home here in Miami. It will be a special match due to everything Newell's Old Boys means to me," said the
Inter Miami coach. "It will also be a good opportunity to prepare for what will surely be an exciting season."

Messi signed for Barcelona after spending six years as a youth player at Newell's, during which time he reportedly scored 234 goals.

Will Inter Miami face River Plate in Dallas?

In late November, it was reported that Inter Miami have arranged to contest a friendly against Argentine giants River Plate in Dallas, Texas in late January.
</item 8>

<item 9>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
@fccincinnati goalkeeper @roman_413 had nothing but praise for @LuchoAcosta94 and Pat Noonan 🟠🔵 pic.twitter.com/Xyp2ozr7n5 — CBS Sports Golazo ⚽️
(@CBSSportsGolazo) October 3, 2023

1 hour 15 mins to kick: Lionel Messi was nominated for not only MLS Newcomer of the Year for the 2023 season, but also 2023 MLS MVP.

Laughable doesn't even begin to describe that blatant PR move, considering that most of the damage Messi has done for Inter Miami has come in Cup competitions.
He has one goal and two assists in 247 minutes of play in actual MLS competition. Lionel Messi has been fun, but let's hope the league doesn't make a mockery of
itself by actually giving Messi either honor.
</item 9>

<item 10>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
🤔 pic.twitter.com/Svq79fvbBG — Major League Soccer (@MLS) October 4, 2023

1 hour to kickoff: With lineups not yet released, there is finally a report regarding Lionel Messi's availability. According to Michelle Kaufman of the Miami
Herald, Messi "did not arrive with the team" to Soldier Field. While that's not outright confirmation that he will not take part in the match, it's a very
strong indication that he will not be involved tonight.

1 hour 15 mins to kick: With lineups due imminently, and nothing to indicate if Lionel Messi is involved tonight or not, the Chicago Fire media team has jokes.
</item 10>
=== END ITEM id=302a138eb0cfe227 ===

=== ITEM id=a698f396b89471c6 ===
Gold sentences and facts:
Sentence 0: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 0.0: The last roundup reported that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 0.1: The last roundup reported that Google tried to have Chrome preinstalled on iPhones.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The person referred to as 'she' admits that she may have misinterpreted the evidence.
  Fact 1.1: The person referred to as 'she' maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers and ad revenue.
  Fact 2.3: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
The DOJ argues Google’s default-search deals and platform control suppress competition and deprive rivals of data; Google says users prefer its superior
product. Trial revelations included $26.3 billion spent on defaults in 2021; Android agreements requiring app preloads and, separately, default Search/Chrome;
and lucrative queries such as iPhone, insurance and travel. Expedia said Google ads became costlier while traffic stagnated as Google promoted its own travel
listings. Evidence suggested a €5 billion EU fine prompted search
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 4>

<item 5>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 10>
=== END ITEM id=a698f396b89471c6 ===

=== ITEM id=404e631f67682613 ===
Gold sentences and facts:
Sentence 0: The Springboks took a 12-3 lead after 34 minutes when the All Blacks once again failed to release on their goal line.
  Fact 0.0: The Springboks took a 12-3 lead after 34 minutes.
  Fact 0.1: The Springboks took the lead when the All Blacks once again failed to release on their goal line.
  Fact 0.2: The All Blacks failed to release on their goal line once again.
Sentence 1: Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide.
  Fact 1.0: The All Blacks sensed an opportunity to strike against 14 men.
  Fact 1.1: On a couple of occasions, the All Blacks kicked for the corner out wide.
Sentence 2: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 2.0: This comes despite Argentina beating the All Blacks last year in Christchurch.
  Fact 2.1: Argentina also gave the All Blacks a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Sam Cane – the All Blacks captain – was forced to watch the match from the sidelines for 53 minutes after having his yellow card for a high shot on Jesse Kriel
upgraded to red. His opposite in the number seven jersey, Siya Kolisi, was later shown a yellow card for a high shot on Ardie Savea, but returned.

Cheslin Kolbe – the magnificent winger who sealed their World Cup triumph four years earlier against England – meanwhile couldn’t bare to watch from the
sidelines after being shown a yellow card for a deliberate knockdown in the 73rd minute.

Advertisement

It was that kind of evening, where the margins were tiny and the outcomes almighty.

“There are no ways I can explain it,” Kolisi said.

“I want to give credit to the All Blacks. They took us to the end, they took us to a dark place. It shows what kind of team they are, to fight with a man down
from early in the game.
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=404e631f67682613 ===

=== ITEM id=d3e3a849aaf4c777 ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: Tyreek Hill missed last week's game.
  Fact 1.1: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.2: Tyreek Hill's stated goal is 2,000 receiving yards this season.
  Fact 1.3: Tyreek Hill needs to reach his 2,000-yard goal this season.

Retrieved items:
<item 1>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:43 p.m. — Hill is up to 65 yards on eight catches. No other Dolphins receiver has more than two catches, and Hill now has 879 yards on the season.

Eagles 17, Dolphins 10

9:39 p.m. TOUCHDOWN — Tagovailoa hits Hill in the end zone for a touchdown! Hill was facing double coverage but simply beat both defenders. Miami has the
deficit back to one possession.

9:38 p.m. — Rookie Nolan Smith sacks Tagovailoa with 44 seconds remaining in the half.

9:37 p.m. — The Dolphins are down to the Eagles' 29-yard line after catches by Braxton Berrios and Tyreek Hill. They have a chance to score on back-to-back
possessions, as they get the ball out of halftime.

9:33 p.m. TWO-MINUTE WARNING — Woah!
</item 1>

<item 2>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Fantasy WR PPR Rankings Week 14: Who to start, sit at wide receiver

Rankings based on full-point PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins vs. Titans

2 CeeDee Lamb, Cowboys vs. Eagles

3 Amon-Ra St. Brown, Lions @ Bears

4 A.J. Brown, Eagles @ Cowboys

5 Keenan Allen, Chargers vs. Broncos

6 Justin Jefferson, Vikings @ Raiders

7 Stefon Diggs, Bills @ Chiefs

8 Mike Evans, Buccaneers @ Falcons

9 Michael Pittman Jr., Colts @ Bengals

10 Davante Adams, Raiders vs. Vikings

11 Brandon Aiyuk, 49ers vs. Seahawks

12 Nico Collins, Texans @ Jets

13 Ja'Marr Chase, Bengals vs. Colts

14 Cooper Kupp, Rams @ Ravens

15 Chris Olave, Saints vs.
</item 2>

<item 3>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 4>

<item 5>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 6>

<item 7>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy WR PPR Rankings Week 13: Who to start, sit at wide receiver

Rankings based on PPR scoring formats

Rank Player

1 Tyreek Hill, Dolphins @ Commanders

2 CeeDee Lamb, Cowboys vs. Seahawks

3 Amon-Ra St. Brown, Lions @ Saints

4 Keenan Allen, Chargers @ Patriots

5 Mike Evans, Buccaneers vs. Panthers

6 A.J. Brown, Eagles vs. 49ers

7 Michael Pittman Jr., Colts @ Titans

8 Ja'Marr Chase, Bengals @ Jaguars

9 Jaylen Waddle, Dolphins @ Commanders

10 Tank Dell, Texans vs. Broncos

11 Brandon Aiyuk, 49ers @ Eagles

12 Garrett Wilson, Jets vs. Falcons

13 Calvin Ridley, Jaguars vs.
</item 8>

<item 9>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 9>

<item 10>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 10>
=== END ITEM id=d3e3a849aaf4c777 ===

=== ITEM id=9d8470f2114b295e ===
Gold sentences and facts:
Sentence 0: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 0.0: Altman has been telling investors that he is planning to launch a new venture.
  Fact 0.1: The Information is the source of this report.
Sentence 1: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 1.0: The theory described here is based on the board's language.
  Fact 1.1: The theory described here is based on the way giant tech companies work.
  Fact 1.2: The theory described here is the prevailing theory floating around right now.

Retrieved items:
<item 1>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 1>

<item 2>
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
</item 2>

<item 3>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 3>

<item 4>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 4>

<item 5>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Could Altman have been working with — or against — OpenAI’s patron in secret? If Altman wanted to kill the golden goose by going independent, that might have
activated the board’s fiduciary or otherwise statutory duty. On the other hand, if he was negotiating some other deal, like an acquisition or deeper and more
exclusive integration, it could also have caused the board to bristle, either at the idea itself or at being excluded.

But if Microsoft was as shocked as the rest of us, as one report has it, it could hardly be the kind of high-stakes conspiracy some seem to be hoping for. But
one must assume that Microsoft would say that either way. Even if they’d been working with Altman on some kind of secret plan, they can truthfully say they were
surprised by his firing. (And they “remain committed to our partnership.”)

2. Do they disagree on long-term strategy?
</item 5>

<item 6>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Or what if, and again this is purely speculation, Altman has been secretly pursuing an internal project, perhaps at significant cost, against the advice of the
board and without the necessary safety measures that probably should accompany such research? It sounds a little wild, but firing your CEO like this is also a
little wild.

Some kind of major mismatch in the financial department could be cause for dismissal, but it’s hard to imagine what Altman could have kept from the board and
CTO that would be so damning.

There’s also the possibility that Altman was making personal investments in a way that the board disagreed with. With OpenAI poised to be a kingmaker in the
field, he would certainly be in a position of power. One would think that, as an ideologically driven person already rich beyond belief and at the head of the
world’s leading AI company, Altman would have risen beyond having to do this kind of side deal, or at the very least that scrutiny on him and those close to him
would prevent them. But one can never be sure.

4.
</item 6>

<item 7>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 7>

<item 8>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
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
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Couldn't be a prouder brother. — Jack Altman (@jaltma) November 17, 2023

It’s unclear what missteps Altman might’ve made helming OpenAI… at present. But they evidently relate to his relationship with OpenAI’s rather unusual board
makeup and corporate governance structure — and perhaps OpenAI’s active talks to raise substantial new capital.

As a recent piece in VentureBeat explored, OpenAI’s for-profit subsidiary, OpenAI Global, LLC, is fully controlled by OpenAI’s nonprofit. While the for-profit
subsidiary is permitted to commercialize its tech, it’s subject to the nonprofit’s mission: attaining artificial general intelligence (AGI), or AI that can — as
OpenAI defines it — “outperform humans at most economically valuable work.”

The nonprofit OpenAI’s board has the power to both determine when the company has achieved AGI and exclude this AGI from IP licenses and other commercial terms,
including with Microsoft, one of OpenAI’s biggest investors and a steadfast integrator of OpenAI’s various technologies.
</item 10>
=== END ITEM id=9d8470f2114b295e ===
