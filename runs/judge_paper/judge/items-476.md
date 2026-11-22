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

=== ITEM id=01b04924a66e50da ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies.'
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried was a former billionaire.
  Fact 1.3: Bankman-Fried got his start in finance at Jane Street.
  Fact 1.4: Bankman-Fried got his start in finance after MIT.
  Fact 1.5: After leaving Jane Street, Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.6: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder, Bankman-Fried, acted in good faith.
  Fact 2.2: The defense said Bankman-Fried never meant to commit fraud or steal.
  Fact 2.3: The defense said Bankman-Fried basically got in over his head.

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
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 5>

<item 6>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 7>

<item 8>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 10>
=== END ITEM id=01b04924a66e50da ===

=== ITEM id=be1944e723fc8899 ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL in the last few years.
  Fact 0.1: Davidson said the level of performance showed the player could control a game and dominate a game in junior.
  Fact 0.2: Davidson said this performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala were among the future NHL stalwarts guided there.
  Fact 1.2: Many other future NHL stalwarts besides those named were also guided there.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“My name’s Connor,” said the other, standing shorter and extending his.

“All right, well we’re just going to go out and I’m going to do an evaluation on you and do some shooting and have some fun, and I’ll make suggestions but I’m
not here to change anything,” Turk said, giving the two boys his usual spiel and thinking nothing of it.

That evaluation is what he calls an “NHL protocol observation assessment,” which runs new clients through specific drills that allow him to visualize their
shooting actions from a technical standpoint (he calls each player’s shot “like a fingerprint”).

That starts with Turk throwing some pucks down in the slot and simply asking them to shoot them stationary so he can watch their body formation, hip and
shoulder positions, spine angles and puck preparation and positioning, paying particular attention to where the puck starts in their stance versus where it
finishes.

Advertisement

On that day, Nate (Danielson, the Red Wings’ 2023 No.
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.
</item 2>

<item 3>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.
</item 5>

<item 6>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 7>

<item 8>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.
</item 9>

<item 10>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 10>
=== END ITEM id=be1944e723fc8899 ===

=== ITEM id=882f428bb8bd309b ===
Gold sentences and facts:
Sentence 0: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 0.0: The case will attempt to argue that Google restrains competition within two separate markets.
  Fact 0.1: One of the two markets is the distribution of apps to Android users.
  Fact 0.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 1: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 1.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 1.1: The three categories are relevance, distance, and prominence.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
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
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
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
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 10>
=== END ITEM id=882f428bb8bd309b ===

=== ITEM id=2e18f594276837b3 ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answer right back and sack Dobbs.
  Fact 0.1: Sweat and Brisker are in there to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 1>

<item 2>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
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
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 4>

<item 5>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
</item 6>

<item 7>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 7>

<item 8>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=2e18f594276837b3 ===

=== ITEM id=e20bd1b891ba82ed ===
Gold sentences and facts:
Sentence 0: But by the age of 16, she is public property – incapable of going outside without being mobbed, and her love of singing and dancing now a lucrative resource.
  Fact 0.0: By the age of 16, the person described is public property.
  Fact 0.1: By the age of 16, the person described is incapable of going outside without being mobbed.
  Fact 0.2: By the age of 16, the person described's love of singing and dancing is now a lucrative resource.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker described it as almost a very strange, subtle clue to the media.
  Fact 1.1: The clue signals that the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker says something really major and traumatic could have happened to her that the media does not know about.
  Fact 1.3: The speaker told NPR this.

Retrieved items:
<item 1>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Only this time, she got to say everything she wanted.

Image source, Universal Music Group Image caption, The video for Shake It Off was filmed under intense secrecy

6) Shake It Off

Rattling like a bag of spanners, Shake It Off is a 21st Century Hey Mickey, a honking great party record with a chorus that drills so far into your brain you
start seeing lights.

It was the last song recorded for 1989, and Taylor arrived at the studio with a challenge for her co-writers Max Martin and Johan "Shellback" Schuster: "We need
drums that make even the person who's having a terrible night at the wedding [go], 'Oh, it's my song!'"

Like Blank Space, Shake It Off is a response to celebrity takedown culture. Taylor lists all of her supposed flaws ("She's an airhead!' 'She dates too many
men!") with a defiant "so what?
</item 1>

<item 2>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's so weird to look back on that - you've got heaping piles of discarded songs that I really loved when I wrote them and I still love now.

"I just need each album to have its own sound and if it doesn't fit that sound, it doesn't matter if it's a great song to me."
</item 2>

<item 3>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I would never have said anything like that on a previous album," she told Ryan Seacrest.

"My previous albums have always been, 'I was right, you were wrong'. What happens when you grow up is you realise that the rules in a relationship are very
blurred. It gets very complicated, very quickly."

4) Out Of The Woods

One of 1989's strengths is how the music reflects the emotional tenor of the words. Here, that means frantic anxiety.

As Taylor and her partner run from forces that seek to destroy them, they're pursued by pounding drums and rushing synths. And there's a desperation to her
voice, as though she's repeating "everything's fine, everything's fine, everything's fine," against evidence to the contrary.

By the middle 8, it seems they've made a clean getaway. Then car brakes are slammed, and it's "twenty stitches in a hospital room".

In interviews, Taylor delighted in revealing this was based on a true story.
</item 3>

<item 4>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Like, through thick and thin, sickness and health love," she told MTV.

Drawing directly on conversations she'd had with Lena, the song blends scenes of everyday life (burnt toast, borrowed shirts) with moments of incandescent
tenderness (kisses on sidewalks and late night heart-to-hearts).

In an interview with Elle, Taylor talked wistfully about the couple's ease around each other.

"I've never had that," she said. "[But] now I have more of a grasp on the fact that when you're in a state of infatuation and you think everything that person
does is perfect, it then - if you're lucky - morphs into a real relationship when you see that that person is not in fact perfect, but you still want to see
them every day."
</item 4>

<item 5>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 5>

<item 6>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
All proceeds from the track are donated to New York City's Public Schools.

Image source, Getty Images Image caption, The original version of 1989 won three Grammy Awards, including Album of the Year

2) Blank Space

Pretty much the perfect Taylor Swift song, Blank Space takes every horrible rumour that was being spread about her love life, and amplifies it with satirical
relish.

"In the last couple of years the media have had a wonderful fixation on painting me as the psycho serial dater girl," she told the Grammy Museum. "Every article
was like, 'Here's Taylor Swift standing near some guy. WATCH OUT GUY!'

"My first reaction was to be like, 'That's a bummer. This isn't fun for me'.
</item 6>

<item 7>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 7>

<item 8>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.
</item 8>

<item 9>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
As the restrictions get tighter, Britney acts out – and ends up legally denied her freedom for 13 years, under a conservatorship controlled by her father.

Her body felt like public property

From day one, Britney’s body is made fair game. Between the videos for her debut single, Baby, One More Time and Sometimes, the media noted her changing
physique and concluded that the 16-year-old must have had breast implants – rather than simply going through puberty. She notices that talkshow hosts ask her
about her chest, while her boyfriend Justin Timberlake gets asked serious questions about his music. It’s one of many pressures that ultimately push her to
breaking point, and when she writes about shaving her head in a salon in 2007, she frames it as a “fuck you” to the impossible beauty standards she felt forced
to live up to: “I’d smiled politely while TV show hosts leered at my breasts, while American parents said I was destroying their children by wearing a crop
top.”
</item 9>

<item 10>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"If she's got blue eyes, I will surmise that you'll probably date her."

The fact that all the new songs were co-written and produced by Jack Antonoff suggests there may have been rights issues over the tracks she wrote with Max
Martin - who has not been involved with the new re-recordings.

That means the vault tracks are all bathed in the dreamy, soft-focus atmospherics of Out Of The Woods, rather than the upfront pop of Shake It Off.

Speaking in 2015, Taylor admitted she'd been brutal about pruning songs from 1989's original tracklist.

"There were so many songs I wrote for this album that were really good, but if I felt in any way that they would have belonged on my last album, Red, they got
cut," she told New York radio's The Todd Show in the Morning.

"There are really, really good songs on the cutting room floor because they weren't innovative enough or new enough.
</item 10>
=== END ITEM id=e20bd1b891ba82ed ===

=== ITEM id=3f7d1e158b650160 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: The Google deals will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
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
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 4>

<item 5>
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
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=3f7d1e158b650160 ===
