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

=== ITEM id=b5f2e1b244a15c6c ===
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
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
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
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 5>

<item 6>
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
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
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
=== END ITEM id=b5f2e1b244a15c6c ===

=== ITEM id=92fc01a6d1b78873 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: A friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expresses his interest in Swift.
Sentence 1: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 1.0: Swift has the Eras Tour booked all through next year.
  Fact 1.1: Swift would rather die than cancel a show.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has now shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 2>

<item 3>
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
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 6>

<item 7>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 7>

<item 8>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
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
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 10>
=== END ITEM id=92fc01a6d1b78873 ===

=== ITEM id=b69db00d6370c174 ===
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
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.
</item 1>

<item 2>
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
</item 2>

<item 3>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 3>

<item 4>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 5>

<item 6>
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
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
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
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 10>
=== END ITEM id=b69db00d6370c174 ===

=== ITEM id=6bdba906a023aad7 ===
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
</item 2>

<item 3>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
Christian Walker had a chance to turn two but couldn't hold onto the ball. The Rangers have the bases loaded with one out.

8:59 p.m. — Lowe keeps the line moving and laces a single up the middle. The Rangers have two runners on with one out. This one could get even more out of hand
for Arizona. Kyle Nelson's night is done.

8:58 p.m. — Jung hits a one-out single for his second hit of the night.

8:53 p.m. — Thomas flies out to left to end the inning.

8:51 p.m. — Gurriel hits a two-out single to give the Diamondbacks a pulse. That'll bring up Alek Thomas.

8:49 p.m. — Heaney notches two quick outs to start the bottom of the second.

8:44 p.m. — The inning ends on a nice throw by Emmanuel Rivera, but what an inning it was for Texas with five runs.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
The Rangers found themselves one strike away from a championship twice in 2011. They couldn't finish the job. On Wednesday night, they'll have a chance to
avenge that painful finish.

Texas found out before Tuesday's Game 4 that slugger Adolis Garcia's season was over, but a talented lineup didn't miss a beat. The Rangers erupted for five
runs in the second inning and five more in the third, bludgeoning Arizona's bullpen to take a 10-0 lead in the blink of an eye.

Corey Seager came through with his third home run of the series, while Marcus Semien drove in five of the Rangers' first 10 runs — including a three-run blast
that put Texas up by double digits.

FAGAN: Three things Diamondbacks need to flip script on World Series

The Diamondbacks have earned a reputation for not backing down, and to their credit, they didn't.
</item 7>

<item 8>
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
</item 8>

<item 9>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 3 (Sporting News, 2023-10-30T22:00:02+00:00)
: Top of 3rd — Corey Seager rips a two-run home run! The Rangers produce some two-out magic as Seager smokes an inside changeup 421 feet over the right field
wall to give Texas a 3-0 lead.

8:41 p.m.: Top of 3rd — Marcus Semien comes up clutch with a single to left center! The star 2B has been struggling this postseason but he delivered here,
driving in Nathaniel Lowe to give the Rangers a 1-0 lead.

8:39 p.m.: Top of 3rd — Brandon Pfaadt struck out Josh Jung swinging and got Leody Tavares to ground out to second but Nathaniel Lowe advanced to third. He is
90 feet away with two outs and Marcus Semien is coming to the plate.

8:36 p.m.: Top of 3rd — Nathaniel Lowe gets the Rangers going right away with a double to right center field.
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
=== END ITEM id=6bdba906a023aad7 ===

=== ITEM id=27b7458801746f60 ===
Gold sentences and facts:
Sentence 0: The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive thinking.
  Fact 0.0: The pastor explained that Jesus brought more positivity than anyone.
  Fact 0.1: The pastor said it is important to keep faith aligned with the practice of positive thinking.
Sentence 1: He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically.
  Fact 1.0: The pastor added that when a person consistently taps into the habit of positive thinking, their relationship with Jesus grows deeper and deeper.
  Fact 1.1: The pastor said the growth in that relationship happens automatically.

Retrieved items:
<item 1>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Bradley said, "Positive thinking is relational, and there is no greater source of joy in our lives than God."
</item 1>

<item 2>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
He said turning to Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.

"The Word of God is living and active, bringing strength, courage, wisdom, and protection," he said.

"God's thoughts and ways are higher than ours, but when we spend time, each day reading His Word and listening to the voice of the Good Shepherd, we are filled
with new thoughts that change the trajectory of our lives."

Bradley also shared his dos and don'ts of positive thinking.

He said it's not "denial and pretending that there are no challenges," he said.

"Positive thinking is not flattery and frivolous compliments."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Rather, positive thinking is realistic, full of truth and love — and aligned with heaven and scripture.

"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating unrealistic expectations for other people and
putting extra stress on them to be perfect."
</item 2>

<item 3>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
A Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your opportunities to overall life
satisfaction, positivity makes a difference."

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

The pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental health.

Bradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from. He said
finding hope during that time was challenging but is how he changes his "hopeless first thoughts."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain," he said.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'
</item 3>

<item 4>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Reject and replace!"

He gave this list of examples:

Replace "People just slack and gossip at my job" with "I will work with all my strength, even when no one is looking"

Replace "There's no hope for my future" with "I am thankful for today and will keep doing the next right thing"

Replace "My children are annoying" with "My children are a gift from God"

Replace "I'll never have good friendships" with "I will pursue and support the people that I respect the most"

Replace "No one really cares about me" with "God is with me and loves me every day"

Said Bradley, "Johns Hopkins reports that there are links between positive thinking and health. People with a positive outlook are less likely to have a heart
attack or cardiovascular event."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain."

Also, "Research from the University of North Carolina points to positive people being more resilient and finding more solutions to problems.
</item 4>

<item 5>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
We do not have to believe, harbor or entertain negative first thoughts. We can be proactive by intentionally choosing our second thought."

It is no accident that the pastor's new book, his second, is called "The Power of the Second Thought."

It dives into the power of understanding that the first thoughts we have are not always the best — and that it's wise to take some time and get to the "second
thoughts," which can yield greater understanding.

STORYTELLER REVEALS HIS FAITH AT WORK: ‘I REFUSE TO TRY TO SCREW SOMEONE IN THE NAME OF BUSINESS’

Said Bradley to Fox News Digital in emailed comments this week, "How can we choose a positive attitude and approach during the day? Reject and replace!"
</item 5>

<item 6>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Positive thinking is not creating unrealistic expectations for other people and putting extra stress on them to be perfect."

Bradley said, "Positive thinking is a vital aspect of loving your neighbor, nourishing your mind, discerning between light and darkness, good and evil and right
and wrong."

"Positive thinking leads to taking action and improving situations. Positive thinking brings out the best in the people around you. Positive thinking rejects
excuses, takes responsibility, and overcomes setbacks."

The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive
thinking.

CLICK HERE TO GET THE FOX NEWS APP

"Jesus exudes a trustworthy positivity that is full of meaning, faithfulness, and transformation," he noted.

"God renews our thinking and vision, empowers us to make intentional choices about our focus and perspective and trains us how to win the battle of our minds,"
he added.
</item 6>

<item 7>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
CLICK HERE TO GET THE FOX NEWS APP

So it's challenged me on that level.

But as it challenges me, it also allows me to go further in that relationship — and to deepen my prayer life.
</item 7>

<item 8>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Pastor Jesse Bradley of Auburn, Washington runs Grace Community Church.

He's ministered to many, many people struggling with a variety of faith dilemmas and issues in their lives.

In this month of October 2023, as Americans grapple with a variety of issues, he addressed the notion of positive thinking — and shared with Fox News Digital
what it would "look like to change your thinking and shift in a new direction."

WASHINGTON PASTOR, ONCE GRAVELY ILL, SHARES HOW ‘THE POWER OF THE SECOND THOUGHT’ TRANSFORMED HIS LIFE

Said Pastor Bradley, "Psychologists report that we have in between 6,000 and 60,000 thoughts a day, based on the person and situation. The National Science
Foundation shares that up to 80% of our first thoughts are not helpful, even destructive."

What happens is that "when life is difficult, we can become flooded with thoughts that are drenched in fear, worry, stress, selfishness, impurity, selfishness
and lies.
</item 8>

<item 9>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
FIRST SUNDAY OF ADVENT IS A REMINDER TO ‘WATCH’ AND PREPARE FOR CHRIST'S SECOND COMING, SAYS FAITH LEADER

There are thousands of prayers on this app. And I'm instantly transported into a sacred space, no matter where I am.

No matter what, if you live in a major city or if you live in a place that's just really busy, you can pop on your headphones and connect to the app. You start
praying with a meditation and you're instantly just transported.

This is something that I've been so grateful for — and especially as we go into the hubbub and the busyness of the Christmas season, it's something that I'm
relying on daily, to be able to just cut through all that noise.

‘Transformative for people’

Question: Are people reaching out to you about your involvement with the app and asking you more about it?
</item 9>

<item 10>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
HALLOW APP ANNOUNCES COLLABORATION WITH LIAM NEESON FOR NEW ADVENT SERIES THIS YEAR

People can "disconnect from the chaos of the real world, the divisiveness, the anxiety" that exists, said Roumie, by connecting with Jesus through prayer.

They can find "peace and a deeper relationship with Christ, and I think that's really the goal of the prayer challenge," he said.

Roumie said that building a habit of prayer "is easier than people think."

He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically. And as a result, you
then experience much more peace, tranquility and serenity in your life and the ability to really muscle through some of the most difficult obstacles and
challenges that life can throw at you," he said.
</item 10>
=== END ITEM id=27b7458801746f60 ===

=== ITEM id=fad84b334327af27 ===
Gold sentences and facts:
Sentence 0: One of the major sources of Gen Z’s distress, of course, is climate change.
  Fact 0.0: Climate change is one of the major sources of Gen Z's distress.
Sentence 1: In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers.
  Fact 1.0: We need to improve our understanding of the drivers of sea surface temperature patterns.
  Fact 1.1: Improving the understanding of drivers of sea surface temperature patterns is the first step.
  Fact 1.2: One driver of sea surface temperature patterns is decadal coupled variability.
  Fact 1.3: Another driver is the pace of ocean heat uptake.
  Fact 1.4: Another driver is the spatial structure of ocean heat uptake.
  Fact 1.5: Another driver is forcing by aerosols.
  Fact 1.6: Another driver is forcing by greenhouse gases.
  Fact 1.7: Another driver is forcing by volcanoes.
  Fact 1.8: We need to improve our understanding of the relative timing of these drivers.

Retrieved items:
<item 1>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.
</item 1>

<item 2>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Roughly 60% of Gen Zers say they are stressed about money this year more than last year, according to a Bankrate survey from July. It’s no surprise either—85%
of Gen Zers say that they couldn’t afford one month’s expenses if they lost their job today.

And since young people are typically affected by inflation the most, as they are the most likely to work part-time or low-paying jobs, this economic climate may
have left Gen Z with permanent “psychological scars,” one expert says.

“How can young people build careers or wealth if they don’t have jobs, and prices of goods and services continue to increase?” Dayo Abinusawa, founder of
London’s Awa Business School and a former lecturer at Cambridge University’s Judge Business School, previously told Fortune.

One Fidelity survey backs this argument: 45% of 18- to 35-year-olds “don’t see a point in saving until things return to normal.”
</item 2>

<item 3>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6 insurrection—an event unseen for centuries in American politics. Then there’s the first major European ground war since World War II in Ukraine, not to
mention market crashes in 2008 and 2020 that recall the Great Depression itself. As the first digitally native generation, Gen Z is experiencing it all through
videos, images, and articles online, which is shaping their mental health, workplace attitudes, and financial habits in visible ways.

It’s no wonder, this Gen Z reporter notes, that 46% of young workers ages 18 to 26 say that they are regularly so distraught over what is happening in the news
that they are unable to function at work, according to a 2023 Edelman report. By comparison, 38% of millennials, 24% of Gen Xers, and 19% of baby boomers and
older generations say the same.

Everything about their behavior communicates that Gen Z is just not okay with it.
</item 3>

<item 4>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6, 2021, following the election defeat of former President Donald Trump. Then there are school shootings, which have only increased in frequency since Columbine
in 1999, with more students documenting the terror on their phones and sharing it online. One of them was the Parkland shooting of 2018, which tragically
created the first spokespeople of the post-millennial generation.

Gen Z’s mental health

Gen Z has the worst reported mental health of any generation—45% of young people report having “excellent” or “very good” mental health, according to a 2018
report by the American Psychological Association.

One of the major sources of Gen Z’s distress, of course, is climate change. Nearly seven in 10 Gen Zers say they experience anxiety when viewing climate change
content on social media, according to a 2021 Pew Research report.
</item 4>

<item 5>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
workforce, but they’re already radically redefining the meaning of work.

Gen Zers want a sense of purpose, so they prioritize environmental, social, and governance (ESG) in the workplace, which encompasses their companies taking
accountability for their sustainability and environmental impacts, providing education and awareness for social issues, and ensuring diverse and inclusive
boards and teams.

And it tracks: Roughly two-thirds of Gen Zers say they frequently speak about important societal issues while at work, according to the Edelman report. They’re
also influencing their older coworkers when it comes to areas like work-life balance, fair pay, and employers’ involvement on social issues.

Gen Z’s financial stress

But don’t forget, Gen Zers have also lived through a global pandemic that shuttered the world for nearly two years, two recessions, and a mounting student debt
crisis, leaving them with little savings but an abundance of financial despair.
</item 5>

<item 6>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
That’s why Gen Z is more concerned with sustainability than any generation before them. Just look at Greta Thunberg: The 20-year-old has become one of the
best-known environmental activists, famously speaking at the United Nations in 2019 with scathing words for world leaders:

“You have stolen my dreams and my childhood with your empty words. And yet I’m one of the lucky ones,” Thunberg said. “People are suffering. People are dying.
Entire ecosystems are collapsing. We are in the beginning of a mass extinction, and all you can talk about is money and fairy tales of eternal economic growth.
How dare you!”

Gen Z in the workplace

By 2030, Gen Z will account for nearly one-third of the U.S. workforce, but they’re already radically redefining the meaning of work.
</item 6>

<item 7>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Everything about their behavior communicates that Gen Z is just not okay with it. This ranges from their widespread, hell-bent determination to find purpose in
work and pushing their employers to have a social conscience, to a sense of despair over their own and the world’s future finances. They have largely given up
on saving money and instead dish out for little “treats” as a way to cope with the larger absurdity of 21st-century life.

Consider the lifetime that was three years ago, as Gen Z emerged into young adulthood, when online videos of the murder of George Floyd shook the country in May
2020, resulting in a summer of violent Black Lives Matter protests and riots. A year later, people watched as armed right-wing extremists stormed the Capitol on
Jan. 6, 2021, following the election defeat of former President Donald Trump.
</item 7>

<item 8>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The inflation bill is responsible for about a quarter of that projected decrease.)

“He brags about how he thinks we should consider him a climate champion because he reentered the Paris accord. That Paris Agreement will take us to about 3
degrees Celsius of global heating,” Kalmus said. “I don’t think we’ll have a civilization at 3 degrees Celsius.”

(A 2022 United Nations report estimated global temperatures would rise between 2.1 and 2.9 degrees Celsius by 2100 if countries held to their climate
commitments. Many countries remain off that pace.)

Climate change is weighing on scientists, but also everyday Americans.

A 2022 poll found almost two-thirds of Americans say they have been affected by extreme weather they believe was at least partially due to climate change.

About 27% of Americans say they are “very worried” about climate change; another 27% just avoid the subject as best they can. One in 10 reported feeling
symptoms of anxiety or depression over climate change.
</item 8>

<item 9>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 9>

<item 10>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial
structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers. An important goal is
to explain historical patterns quantitatively and to predict their likely future evolution. Second, we need to quantify the dependence of local and remote
top-of-the-atmosphere radiative fluxes on the magnitude, spatial scale, and sign (positive or negative) of surface temperature changes. Third, we have to
explain how ocean heat uptake and radiative feedback are connected on various timescales, both globally and regionally.

Interpreting Paleorecords Requires Caution

In the same way that the historical evolution of surface warming and the planet’s radiation budget inform Earth’s future only to a limited extent, analyses of
paleorecords for recent and deep time intervals must account for the pattern effect to be applicable to climate projections of the future.
</item 10>
=== END ITEM id=fad84b334327af27 ===
