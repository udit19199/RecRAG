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

=== ITEM id=a06751c566b595d4 ===
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
</item 1>

<item 2>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.
</item 2>

<item 3>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE
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
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.
</item 8>

<item 9>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The Texas Chainsaw Massacre 2 is nothing like that. You won’t see its name in a film school syllabus, and the films it inspired can be counted on one hand.
Nonetheless, there’s something commendable about the gall of Tobe Hooper, the original Texas Chain Saw director, returning to his series with the energy of a
middle-aged man on his first ayahuasca retreat.

Leatherface gets in a high-speed car battle, eventually standing on the roof of a speeding sedan. Dennis Hopper dual-wields chainsaws. A radio station office is
obliterated like it’s a novelty rage room. Sometimes you want a steak, and sometimes you want a hamburger. Texas Chainsaw Massacre 2 is a damn juicy burger. —CP

Oct. 27: Possessor

Where to watch: Hulu

My first Brandon Cronenberg (yes, David’s son) movie was Infinity Pool, his latest messed-up meditation on the rich and powerful.
</item 9>

<item 10>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
An unsettling tale about a plastic surgeon (played by Pierre Brasseur) who kidnaps young women and performs surgery on them to try and find a face replacement
for his daughter (Édith Scob), Eyes Without a Face is equal parts haunting and beautiful. Scob’s iconic face mask in the movie was later referenced in her role
in the also-excellent Holy Motors many decades later. —PV

Hellraiser

Director: Clive Barker

Cast: Andrew Robinson, Clare Higgins, Ashley Laurence

Where to watch: Prime Video, AMC+, Shudder, Tubi, Pluto, Hoopla

Clive Barker’s 1987 directorial debut adapts his 1986 novella The Hellbound Heart to tell the story of Larry (Andrew Robinson) and Julia Cotton (Clare Higgins).
The Cottons are a married couple who move into the home of Larry’s recently deceased brother, Frank (Sean Chapman), with whom Julia had a previous affair.
</item 10>
=== END ITEM id=a06751c566b595d4 ===

=== ITEM id=faabffa7b4840d18 ===
Gold sentences and facts:
Sentence 0: After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry at his family’s response to it.
  Fact 0.0: William learned of Diana’s death from his father.
  Fact 0.1: William learned of Diana’s death at the royal residence of Balmoral in Scotland.
  Fact 0.2: The character William is played by Rufus Kampa.
  Fact 0.3: William is left heartbroken with grief.
  Fact 0.4: William is angry at his family’s response to Diana’s death.
Sentence 1: The latest installment takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
  Fact 1.0: The latest installment takes place between the late 1990s and the mid-2000s.
  Fact 1.1: The latest installment includes the lead-up to Princess Diana’s fatal Paris car crash.
  Fact 1.2: Princess Diana’s fatal Paris car crash happened in 1997.

Retrieved items:
<item 1>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“The last call between Harry, William and Diana is heartbreaking,” a third user commented.

In both the series and in real-life, Diana and Dodi - along with their chauffeur, Henri Paul - were killed on 31 August 1997 when their Mercedes-Benz was
pursued by paparazzi and crashed in the Pont de l’Alma tunnel in Paris. The mother of two was 36 years old.

The fatal car crash was recreated in the very first scene of season six. While the incident is not shown on-screen, a car carrying Diana and Dodi is seen
speeding past a man walking his dog, before entering the tunnel. As the vehicle’s horn blares, he calls emergency services on his cell phone.

Australian actor Elizabeth Debicki, who took over the role of Diana from Emma Corrin in season five, admitted it was “difficult to recreate” the scene -
especially with her character being hounded by the paparazzi. “It was heavy and very manic, and incredibly invasive.
</item 1>

<item 2>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
He is not located, but later walks back to the house on his own accord, drenched in rain. “14 hours, that poor boy was gone,” the Queen later says.

William (Kampa) goes missing after his mother’s death (Keith Bernstein)

It’s true that William and Harry initially stayed in Balmoral with their grandmother following the death of Diana. While it’s not known whether he went missing,
during a visit to Scotland in 2021, William recalled how that experience was one of the “saddest” moments of his life. “I was in Balmoral when I was told that
my mother had died. Still in shock, I found sanctuary in the service at Crathie Kirk that very morning,” he recalled. “And in the dark days of grief that
followed, I found comfort and solace in the Scottish outdoors. As a result, the connection I feel to Scotland will forever run deep.”

The Crown season six part one is on Netflix now, with part two arriving on 14 December.
</item 2>

<item 3>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Dodi’s proposal to Diana on the night they died

After bursting into tears going for dinner at the Ritz, move upstairs, he proposes to her with a ring she had previously pointed out when they sheltered from
crowds in a Monte Carlo jewellery shop. She tells him to get up off his knee “I know the whole world is wondering if we’re going to get married, but that’s not
a reason to actually do it.” However, they share a drink and bittersweet conversation about how they need to live their lives going forward - advice

Dodi is shown proposing to Diana before they died (Daniel Escale/Netflix)

In real life, Dodi did not propose to Diana on that fateful evening in Paris. However, at the inquest into the death in 2007, the jury were shown CCTV footage
of him purchasing an engagement ring worth £11,600 in a jewellers across the square from the Ritz on the afternoon of the crash.
</item 3>

<item 4>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
It was later believed to have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also
the name of episode three – was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed
claimed that the pair had met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.

Prince William goes missing in Balmoral after Diana’s death

After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry
at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find him”, prompting Charles and
Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.
</item 4>

<item 5>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
The season begins with Princess Diana (played by Elizabeth Debicki) whisking her sons - Prince William (played by Rufus Kampa) and Prince Harry (played by Fflyn
Edwards) - away on summer holiday to Saint-Tropez, France. After staying several days on Mohamed Al Fayed’s yacht, the boys return home to London where their
father, the then-Prince of Wales, accompanies them to Balmoral Castle to vacation with the rest of the royal family in Scotland.

In episode three, Diana’s partner Dodi Fayed (played by Khalid Abdalla) surprises her with a quick trip to Paris, France. However, the princess is keen to
return home to the United Kingdom, and is eager to speak to her sons over the phone. A mob of paparazzi following the couple delays Diana’s scheduled phone call
with William and Harry, until she’s able to catch them again later that evening.

It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.
</item 5>

<item 6>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

The first four episodes of The Crown season six are officially here, but many viewers believe one scene has depicted the most devastating moment throughout the
entire series.

On 16 November, Netflix dropped the first four episodes of The Crown’s final season, followed by the last six episodes on 14 December. The latest installment
takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
</item 6>

<item 7>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Through tears, she claimed that he had proposed to her (something his family denied) and promised to buy her a house in Malibu. However, she dropped the lawsuit
following Dodi’s death.

Diana’s close relationship with paparazzi on France yacht trip

When Dodi and Diana first spend time together on his father’s yacht in France, the paparazzi are never far away. However, when William refuses to go outside
until the photographers leave them alone, she takes a boat out to visit them. Wearing a swimming costume, she asks them: “We’re having a lovely time, apart from
one little thing: you lot.” “Don’t be like that, you love us really,” they reply. Diana asks them to leave her alone as they’re “freaking out” her sons, before
telling one photographer – who she knows by name – that if they go away, “you’re going to get a big surprise with the next thing I do”.
</item 7>

<item 8>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It’s just a bit mad here. I don’t really understand how I ended up here. Mummy just needs to make some changes to her life, that’s all. But that’s not your
problem, that’s mine.”

Before saying their goodbyes, Diana reminds William and Harry that she will be seeing them the next day - although that day never comes. “I’ll see you both
tomorrow night,” she says. “Just the three of us.”

The moment unsurprisingly tugged on the heartstrings of many viewers, who took to social media to share their thoughts on the devastating moment.

“The last phone call between Harry, William and Diana broke my heart,” one fan wrote on X, formerly known as Twitter.

“That precious phone call between Diana and her sons,” said someone else. “Don’t mind me, I’m just gonna cry in one corner.”

“The last call between Harry, William and Diana is heartbreaking,” a third user commented.
</item 8>

<item 9>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 9>

<item 10>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
With the show heading into the late Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being
depicted on screen that many of the public will remember well.

The first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his relationship with Camilla Parker Bowles
(Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid Abdalla). The show also features the
pair’s death in a car crash in Paris.

As the new season arrives, and fans wonder what in The Crown is based in reality, here’s everything you need to know.

Diana’s role with Tony Blair’s government

Episode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia Leonard) with William after asking for a meeting with him.
</item 10>
=== END ITEM id=faabffa7b4840d18 ===

=== ITEM id=c2c5f9342b0a38ca ===
Gold sentences and facts:
Sentence 0: UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.
  Fact 0.0: NFL Network's Ian Rapoport reported on a Sunday morning.
  Fact 0.1: Ian Rapoport reported that Hockenson is expected to play against New Orleans.
Sentence 1: Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a chance to play.
  Fact 1.0: Head coach Dennis Allen told reporters on Friday.
  Fact 1.1: Dennis Allen said Olave was a full participant in Friday's practice.
  Fact 1.2: Dennis Allen said Olave is still in concussion protocol.
  Fact 1.3: Dennis Allen said Olave has a chance to play.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 1>

<item 2>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Meanwhile, Pipkins was also limited on Friday.

The Patriots will be down a couple of wide receivers as Douglas and Boutte have been ruled out after not practicing on Friday. For Douglas, he missed the entire
week due to a concussion he suffered against the Giants, while Boutte was downgraded on the injury report on Thursday and missed back-to-back days due to an
illness on top of his shoulder injury.

Lions (-4) at Saints

The Lions just designated Hooker for return, who has been rehabbing a torn ACL he suffered at Tennessee last year, this week so the rookie quarterback was not
expected to be activated this week. Meanwhile, Anzalone did not practice at all this week, so his prospects look bleak for Week 13. As for Jackson, he was a
full participant on both Thursday and Friday.

Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a
chance to play.
</item 2>

<item 3>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 3>

<item 4>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.
</item 4>

<item 5>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 5>

<item 6>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 8>

<item 9>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 9>

<item 10>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 10>
=== END ITEM id=c2c5f9342b0a38ca ===

=== ITEM id=10af6fd365f92d8f ===
Gold sentences and facts:
Sentence 0: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 0.0: Wired removed the article for not meeting its standards.
  Fact 0.1: The article was shared widely on social media.
  Fact 0.2: The article was boosted in write-ups at other outlets.
  Fact 0.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 1: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 1.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 1.1: Prabhakar Raghavan revealed that Google paid $26.3 billion in 2021 to multiple browsers, phones and platforms.
  Fact 1.2: The payments went to companies including Apple, Samsung and Mozilla.
  Fact 1.3: The Verge reports this.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content, their readers and ad revenue through anticompetitive means.

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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 6>

<item 7>
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
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 10>
=== END ITEM id=10af6fd365f92d8f ===

=== ITEM id=753b37b65db5548b ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings had a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The team under discussion has seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams seem to know they will not be able to succeed on downfield shots against the team under discussion.
  Fact 1.3: The team under discussion has had zero completions of 20-plus yards from tight ends.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 1>

<item 2>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 4>

<item 5>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 7>

<item 8>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
No one has scored a touchdown yet tonight.

10:40 p.m. — Addison's catch stands, and an Alexander Mattison run has the Vikings at the edge of field goal range. They want more than three points, but it's
early enough that they absolutely could settle for a field goal if it got to that point.

10:37 p.m. — Matt Eberflus throws the challenge flag on a nice grab by Addison to give the Vikings a first down. This one looks like it will be tough to
overturn...

10:34 p.m. — Wow, Jordan Addison would've had a walk-in touchdown, but Dobbs' throw takes him out of bounds. Addison couldn't have been more wide open.

10:33 p.m. — Brandon Powell has been the Vikings' biggest playmaker at two key points tonight. He goes 18 yards here to put Minnesota near midfield.

10:29 p.m.
</item 10>
=== END ITEM id=753b37b65db5548b ===
