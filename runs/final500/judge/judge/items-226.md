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

=== ITEM id=4279ce10178b42fb ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:
<item 1>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 7>

<item 8>
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
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
</item 10>
=== END ITEM id=4279ce10178b42fb ===

=== ITEM id=ca71d0c5742e9030 ===
Gold sentences and facts:
Sentence 0: Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it.
  Fact 0.0: Valve has announced a new Steam Deck.
  Fact 0.1: The announcement of the new Steam Deck was a surprise.
  Fact 0.2: The writing outlet has already reviewed the new Steam Deck.
Sentence 1: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 1.0: Valve made a number of minor physical upgrades with its new Steam Deck iteration.
  Fact 1.1: The minor physical upgrades were made alongside big improvements.
Sentence 2: For now, Griffais explained, it's important to Valve that developers don't need to "split their attention" between different performance targets.
  Fact 2.0: Griffais explained that it is important to Valve that developers do not need to "split their attention" between different performance targets.
  Fact 2.1: Griffais said this is important to Valve for now.
Sentence 3: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 3.0: The Steam Deck OLED will go on sale on November 16 at 1pm ET.
  Fact 3.1: Valve has Steam Deck OLED units ready to ship on November 16.

Retrieved items:
<item 1>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The focus of the M3 chips is enhanced graphical capabilities. Compared to the previous M1 chip, Apple asserts that the M3 renders at 2.5 times the speed and has
a CPU that is 30% faster. The event’s videos looked great (of course they did — they were produced by Apple), but one delightful quirk was that they were all
shot on iPhones, showing off that the phones are becoming more and more capable, even for professional work.

Magic Leap, an augmented reality hardware company, has replaced its CEO Peggy Johnson with Ross Rosenberg, marking the completion of its enterprise pivot.
Johnson joined Magic Leap in August 2020 and led the shift in business strategy. Her successor previously served as executive at Belden and First Solar, and
will now guide the company through its next period, as the company faces the daunting challenge of achieving profitability.

More from hardware land:

Is it a bird? Is it a plane?: Shield AI, a startup specializing in autonomous flying systems for the U.S.
</item 1>

<item 2>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 2>

<item 3>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 3>

<item 4>
Graph query row:
article: The Steam Deck OLED improves much more than its screen
relationship: PUBLISHED_BY
related: Polygon
</item 4>

<item 5>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 5>

<item 6>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 6>

<item 7>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 7>

<item 8>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Security researchers found that the feature, introduced in iOS 14, was not working as intended.

$400 million security acquisition: At a time when the tech industry in Israel is disrupted due to political events, Palo Alto Networks is acquiring Israeli tech
company Dig Security for an estimated $400 million. This acquisition will bolster Palo Alto’s cloud security-focused Prisma business.

I guess we gotta pay up: Paying off hackers in the event of a ransomware attack is pretty common practice, but it comes with a ton of pitfalls: You are trusting
anonymous criminals to do what they promised, and you may potentially be violating U.S. sanctions laws along the way.

What’s brewing on the silicon farm?

Apple’s “Scary Fast” October event saw the announcement of their new M3 chip lineup that includes M3, M3 Pro, and M3 Max. These chips, which were the highlight
of the event, will be integrated into the new 24-inch iMac and the new MacBook Pros.
</item 8>

<item 9>
Graph query row:
article: The Steam Deck OLED improves much more than its screen
relationship: MENTIONS
related: Valve
</item 9>

<item 10>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 10>
=== END ITEM id=ca71d0c5742e9030 ===

=== ITEM id=4a9e35a3d449bb0d ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google “siphons off” news publishers’ content.
  Fact 0.2: The case argues that Google siphons off news publishers’ readers.
  Fact 0.3: The case argues that Google siphons off news publishers’ ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.

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
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 7>

<item 8>
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
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 10>
=== END ITEM id=4a9e35a3d449bb0d ===

=== ITEM id=524cecfac05331a5 ===
Gold sentences and facts:
Sentence 0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for technologists and designers to be in the process together.
  Fact 0.0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies.
  Fact 0.1: In this innovation cycle, there is a strong need for technologists and designers to be in the process together.
Sentence 1: Stufish partner Ric Lipson (along with artist Es Devlin and producer Brian Eno) collaborated on the U2 project for Sphere.
  Fact 1.0: Ric Lipson is a partner at Stufish.
  Fact 1.1: Ric Lipson collaborated on the U2 project for Sphere.
  Fact 1.2: Es Devlin, an artist, collaborated with Ric Lipson on the U2 project for Sphere.
  Fact 1.3: Brian Eno, a producer, collaborated with Ric Lipson on the U2 project for Sphere.

Retrieved items:
<item 1>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Product design is in a moment of profound change and redefinition as technologies like artificial intelligence (AI) and spatial computing dramatically affect
computing experiences. AI, especially, may have only minor impacts on interface design but will significantly impact the holistic product or ecosystem
experience. Spatial computing, on the other hand, will change human-computer interactions and unmoor our understanding of what a computer is.

In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for
technologists and designers to be in the process together.

For successful products and businesses, innovation is perpetual. There is a never-ending search to find the next new thing that enhances the user experience,
extends product reach, expands revenue, or all three simultaneously. Product design makes innovation less daunting and increases chances for success because it
is a multidisciplinary process with structures and frameworks to catalyze innovation. Technologists have a role in the process that expands beyond simply
validating the technology or concept, which is their customary responsibility.
</item 1>

<item 2>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 2>

<item 3>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Technologists must “find the grain” of a technology. In this way, they become an intermediary between the abstract nature of design and the pedantic nature of
technology. This philosophical perspective is especially important when a product is in a growth stage or when using emerging technologies.

Regardless of whether your product is growing or in a stable stage or employs established or emerging technologies, integrating technologists into the product
strategy and design process enriches the final outcome. There is a technology perspective extending beyond the operations and mechanics of the code “factory
floor,” which provokes innovation. Sometimes this leads to small, impactful moments of innovation — and sometimes it is a brilliant revolution.
</item 3>

<item 4>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
For most technical leads and software developers, this is a mode inversion from our traditional tactical and direct interaction with technology. The context
switch from daily build-and-operate is challenging but is paramount to developing successful and innovative products. We are uniquely positioned to generate
strategic insights translated from dense technical minutiae that drive innovative business cases and product experiences.

A technology innovation must solve a business problem, such as improving operational efficiency, growing existing revenue streams, or generating new ones. The
problem space may be customer-facing (e.g., how can we deliver a new feature?) or internal-facing (e.g., how can we make a process more efficient?). The problem
is the paramount concern. The specific technologies or innovations used to solve the problem are often less critical. We cannot lose perspective of the business
needs; otherwise, the activities become too academic or a paid hobby.

An ordinary domestic analogy is hanging a picture. The hole size, bracket, or tools used to hang the picture are immaterial as long as the picture is on the
wall and straight.
</item 4>

<item 5>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
The specifics of the process and technologies are only important as they pertain to how well they solve the problem, the costs to do so, and the overall end
user experience.

Product innovation is experimental and should not always be expected to yield productive results. It requires a learning curve and patience, as the outcomes are
often ambiguous and unknown. Business leadership can struggle with this perspective because it is indefinite (in terms of results and timelines), and it is
challenging to translate pure technology innovation into value creation. A gap opens between technology and product teams wherein technology teams strain to
articulate the capabilities and value of a technology innovation, leading to unfilled promises and the perception of “technology for the sake of technology” or
quips like “a solution looking for a problem.”

The current hype cycle in AI serves as a great concrete example. For technology or product executives, the challenge is how to do more than check the AI box —
it’s about how to meaningfully incorporate AI into a product. Instead of forcing a technology onto a product, the design process flows to the technologies.
</item 5>

<item 6>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Instead of forcing a technology onto a product, the design process flows to the technologies. In this way, the technology becomes a natural solution.

As experts on a technology or technology stack, we can communicate abstract insights or contribute in more conceptual contexts. Technologists add value to the
product design process by sharing their expertise on a technology’s characteristics. Designers use this information to shape and exploit technologies in the
visual and interaction design process. In this way, technologists inform new interaction models, interface metaphors, and product channels. This involvement
gives confidence and conviction to the promises of design.

Think of digital technologies as a material like paint, stone, or wood. For artisans to create with materials, they must understand the material’s ontology and
phenomenology. An artist must know the differences between oil, acrylic, or watercolor paints because each material has different properties that affect how and
what can be created. Technologists must “find the grain” of a technology.
</item 6>

<item 7>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Too often, design is minimized to only the act of making user interfaces pretty. Product design is much deeper and broader in scope than visual design assets.
For example, product design can give direction and focus to a business strategy, user experience strategy, or technology explorations.

The process establishes a guide rail throughout any innovation initiative. The core of product design is decision-making with an astute instinct for making the
best decisions at the most opportune time. Product design helps reduce risk and leads to more effective innovation through quality decision-making.

A progressive role for technologists

Technologists play a strategic role in product innovation and should bring a metaphysical perspective in addition to being punctilious. Our job is to
communicate the essence of a technology and think strategically about the application of technologies to problem spaces. We are most constructive when we
translate the technicals of “how to make X do Y” to “these are the types of products and services realizable with technology X.”
</item 7>

<item 8>
Inside the $US2.3b venue that will transform the concert industry (The Sydney Morning Herald, 2023-10-19T18:00:00+00:00)
Sphere, in contrast, more wholly and subtly envelops you. “They are both legacy projects made by artists who have the career and track record to be able to look
at these things objectively in a way that a younger performer perhaps wouldn’t because it takes an entire lifetime of experience to make those kinds of choices
and make it work,” Williams says. The two shows also share British music stage architecture firm Stufish whose work ranges from the Queen’s Platinum Jubilee,
tours for Beyoncé and Jay-Z, Michael Bublé, Elton John and Madonna as well as the ABBA Arena in London. Stufish partner Ric Lipson (along with artist Es Devlin
and producer Brian Eno) collaborated on the U2 project for Sphere. “With the ABBA Arena, it’s absolutely designed for that show, but you could also imagine that
building having another avatar show in there, or indeed a live performance where people would use the technology that’s in the building as an immersive
environment,” Lipson says.
</item 8>

<item 9>
Inside the $US2.3b venue that will transform the concert industry (The Sydney Morning Herald, 2023-10-19T18:00:00+00:00)
Critically, says Lipson, the staging for the U2 show is elementary in real terms, despite its technological complexity. “And what’s interesting is that the
Sphere, like ABBA Arena, is only touching the sides of what you can do,” he says. “In five years’ time and 10 years’ time, two years’ time, whenever the
technology shifts, you’ll be able to do more.” By the numbers ... U2 perform at the Sphere in Las Vegas. Credit: Stufish Entertainment Architects The screens,
for example, are a bespoke assembly of state-of-the-art screens. But behind those screens are “a whole new kind of speaker,” Lipson says. “In a regular show,
the speakers are at one end pushing the sound down the room, so it’s louder at the stage end and quieter further away. [But here] because everybody’s sort of
getting their own feed, everybody’s getting an equal volume. So hearing the show in there, sonically, may be better than anywhere.”
</item 9>

<item 10>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Before discussing the nontraditional ways technologists participate in product innovation, let’s conceptually examine innovation and product design.

The word innovation has a simple meaning — to introduce something new or a novel method of doing something — yet we frequently inflate it to mean something
magical, grandiose, and world-changing. Innovation can just as easily be mundane and straightforward. The key to innovation is being “new.” The “new” can be
localized to a team, product, process, or business unit. The “novel” can be well-known and established practices not incorporated into your workflow or product.
Sometimes, innovation comes from closing small gaps and isn’t always a grand eureka moment.

Instead of forcing a technology onto a product, the design process flows to the technologies. In this way, the technology becomes a natural solution.

Product design is a process and not a discipline or deliverable. It is (understandably) easy to limit the scope of product design to color choices, content
layout, and aesthetics.
</item 10>
=== END ITEM id=524cecfac05331a5 ===

=== ITEM id=35c3c92afbf03051 ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answered right back.
  Fact 0.1: At 9:47 p.m., the Bears sacked Dobbs.
  Fact 0.2: Sweat and Brisker were in on the sack to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 2>

<item 3>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.
</item 3>

<item 4>
The article assesses two AFC contenders. Baltimore’s offense has scored and generated less EPA through six games than any Lamar Jackson-led Ravens team; his
explosive runs have also declined as the team shifts toward passing. An executive doubts the change will help when it matters, warning that limiting Jackson’s
runs removes the offense’s biggest weapon. Jacksonville’s outlook depends partly on Trevor Lawrence’s knee; the young, scrappy team has growth potential, and
coach Doug Pederson brings championship experience.
</item 4>

<item 5>
The material compiles Sporting News and CBS Sports coverage across sports. It contrasts the NBA’s Christmas audience (4.27 million average) with the NFL’s
record Thanksgiving average (34.1 million), noting preference determines which tradition is “better.” Most items offer 2023 NFL fantasy, DFS, betting, and
lineup advice: rankings, start/sit calls, picks, props, and player stacks, shaped by matchups, injuries, bye weeks, and scoring rules. Other coverage previews
bowl games and
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 7>

<item 8>
After Philadelphia’s loss, executives debated its late third-and-9 interception versus running and punting, highlighting tension between analytics and caution.
The Eagles’ explosive-pass rate has dropped from 21% to 11%, perhaps because defenses are adapting, though league-wide declines and effective runs and short
passes offer context. Baltimore’s offense and Lamar Jackson’s explosive runs have faded as passing increases, potentially sidelining its greatest weapon.
Jacksonville’s growth depends partly on Trevor Lawrence’s knee; coach Doug Pederson brings
</item 8>

<item 9>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 10>
=== END ITEM id=35c3c92afbf03051 ===

=== ITEM id=d7c605d4fb3f4ab9 ===
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
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Broncos completed a major comeback in the Mile High City.

Both Denver and Minnesota entered "Sunday Night Football" on red-hot win streaks but only one team could keep the momentum rolling. It looked like it was going
to be Joshua Dobbs and the Vikings for the majority of the contest, but the Broncos wouldn't let up.

Trailing by eight points going into the final frame, Denver's offense finally woke up in the game's biggest moment. The Broncos hadn't found the end zone all
night but Russell Wilson still found a way to lead an impressive game-winning drive in the final three minutes.

Courtland Sutton was as clutch as can be, coming up with a key catch on fourth down to keep the drive alive. When Denver brought the ball into the red zone
where it had stalled out several times throughout the night, Wilson went right back to his guy Sutton.

Sutton hauled in a monster touchdown reception to give the Broncos a one-point lead with a minute remaining, and Dobbs had run out of magic.
</item 1>

<item 2>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
He's responsible for all of Denver's points as it trails 10-9.

9:35 p.m.: Russell Wilson has the Broncos' offense moving. He's delivered four complete passes for 46 yards, bringing Denver down to Minnesota's 34-yard line
with 11 seconds on the clock. The Broncos have one timeout remaining.

9:27 p.m.: Alexander Mattison was stuffed on a three-yard run, then Joshua Dobbs couldn't find his RB on a swing route on 3rd down, forcing a punt. The Broncos
will start their drive from their own 25 with 1:11 remaining in the half after a touchback on the punt.

9:24 p.m.: Joshua Dobbs connected on a 17-yard pass to Jordan Addison and a 10-yard pass to TJ Hockenson to bring Minnesota to its own 48 at the two-minute
warning. It will be 2nd & 5 coming out of the break.

9:21 p.m.
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 4>

<item 5>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 5>

<item 6>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Three of the seven teams passed, resulting in one first down, one sack and Hurts’ interception. Of these three teams, only the Eagles lost the game.

3. The 49ers lost Sunday when their rookie kicker missed from 41 yards. Is San Francisco wise trusting a championship roster to a rookie’s foot come playoff
time?

The 49ers raised eyebrows among some when they used their second-highest pick in the 2023 draft — a third-round choice, 99th overall — for a kicker. The
selection of Michigan’s Jake Moody gave the 49ers a cost-effective successor to veteran Robbie Gould. General manager John Lynch called Moody a potentially
“foundational” player whose stock in the draft was so high, other teams were trying to trade ahead of San Francisco for a shot at him.

Who do you want lining up the potential winning kick in a Super Bowl?
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=d7c605d4fb3f4ab9 ===
