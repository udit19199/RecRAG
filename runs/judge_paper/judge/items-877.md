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

=== ITEM id=2607fd8c69e096d1 ===
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
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—AG

Oct. 16: Project Wolf Hunting

Where to watch: Hi-Yah!; available to rent on Amazon, Apple, and Vudu

Sometimes, what you’re looking for is just literal tons of blood and gore. And when that’s contained in a premise that’s essentially “Con Air meets Predator,
with a bit of The Last Voyage of the Demeter,” that’s hard to beat.

Project Wolf Hunting is a Korean monster mash built out of an extremely fun idea. “What if a bunch of dangerous criminals rebelled on the cargo ship they were
being transported on, only to discover the ship is also transporting a monster?”

The story is necessarily light, and the characters are thinly drawn (although the effective costuming and makeup work go a long way), but that’s not what you’re
here for if you’re watching Project Wolf Hunting. You’re here for gory violence, and boy, is there a lot of it.
</item 1>

<item 2>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
The possessed-animatronics horror video game Five Nights at Freddy’s is less than a decade old, but a movie version has been in the works almost since the game
launched, which seems especially protracted for a development process. Warner Bros. bought the movie rights in 2015, and the project moved over to Blumhouse a
couple of years later. Game creator Scott Cawthon wrote and rewrote the script, while kid-horror filmmakers like Gil Kenan (Monster House) and Chris Columbus
(Gremlins, the first Harry Potter and Percy Jackson movies) signed on and jumped off the project.

Finally, under director Emma Tammi, the film is hitting theaters simultaneously with a Peacock streaming debut. The franchise’s most hardcore fans probably
consider themselves beyond ready to watch the Freddy’s film. But what about everyone else? If you’re not sure, have a look at this handy (and extremely
spoiler-light) Q&A.
</item 2>

<item 3>
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
</item 3>

<item 4>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—TE

Oct. 24: Overlord

Where to watch: Paramount Plus, Pluto TV

This slept-on genre-mashing gem takes a cast of excellent actors about to do bigger things (Jovan Adepo, Wyatt Russell, John Magaro, Pilou Asbæk, and more), and
an up-and-coming director (The Pope’s Exorcist’s Julius Avery) and places them behind Nazi lines during World War II. The Nazis are up to more than usual this
time: There be zombies in these labs.

On the eve of D-Day, a group of Allied paratroopers are sent into Germany on a covert mission. Things go wrong instantly when their plane crashes, and they soon
discover twisted Nazi experiments after connecting with the residents. Things get bloody fast, with great action from second unit director Shaun O’Dell (The
Woman King, Brotherhood of the Wolf) and stunt coordinator Jo McLaren (Heart of Stone).

Overlord is a gleeful, intentionally ridiculous Nazi-zombie-bashing time, with tight direction and a fantastic group of actors.
</item 4>

<item 5>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
The movie follows a group of friends who find themselves hunted by a family of cannibals in the middle of Texas, and is a chilling, violent fever dream that
permanently lodges itself in the minds of those who watch it.

Eight films have followed, including a Netflix version in 2022, but the original stands out as an unhinged encapsulation of pure chaos and terror. At a tight 83
minutes, the original Texas Chain Saw Massacre is well worth the small time investment to catch up on one of the most influential horror movies ever made. —PV

The Thing

Director: John Carpenter

Cast: Kurt Russell, Wilford Brimley, Keith David

Where to watch: Peacock

John Carpenter’s postmodern creature feature takes the idea of alien monsters and makes them simultaneously more recognizable and more gross and unworldly than
in any other movie in history. The Thing, the second adaptation of the excellent novella Who Goes There?, remains thrilling, terrifying, and absolutely
disgusting more than 40 years after its release.
</item 5>

<item 6>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Thankfully, their helpful hitchhiker companion Baby knows just the place they can get help: the Firefly mansion, where her family lives. From there, things
devolve into a host of horrific stage plays, cursed experiments, and gruesome murders.

The family at its center, whom Rob Zombie went on to make two more movies about, is repulsive and absolutely horrible, but still zany enough to be truly
unsettling in a way that few horror movies can be. House of 1000 Corpses is a grisly, grotesque film that captures all the grime of the ’70s movies that
inspired it. But what really makes it worth watching is the way it balances silliness and terror, doling each out in proportional quantities until you can’t
figure out if you should laugh or scream. Instead, you just end up doing both. —AG

Oct. 30: Thirst

Where to watch: Peacock

Park Chan-wook has never missed.
</item 6>

<item 7>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Evans’ mesmerizing Apostle deliberately starts in roughly the same place as the classic 1973 suspense thriller The Wicker Man, with a lone man heading to a
remote Welsh island after getting a letter suggesting his sister is being held against her will there by a dangerous cult. Worming his way into the community by
pretending to be a convert, Thomas (Dan Stevens, even more feral and intense than he is in The Guest or Beauty and the Beast) inevitably discovers a lot of
nasty work going on on the island.

Evans charts his own course with Apostle, veering far from the Wicker Man mold and into much bloodier territory. But he takes advantage of some of the same
ideas: the ominous isolation of the island, the close-knit secrecy of its community, the predatory ideas that have flourished there under charismatic
leadership. Stevens is particularly terrific in this movie, lunging from one scene to the next like a humanoid wolf who can barely contain his bloodlust.
(Thomas has plenty of closely held secrets of his own.)
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Lovecraft and the anti-consumerist allegories of George A. Romero. The horrors hold up over 50 years after it was first released, and with just the right
combination of inadvertent silliness and legitimate terror, Messiah of Evil is a satisfying watch and an excellent start to a month of horrors. —Toussaint Egan

Oct. 2: Ginger Snaps

Where to watch: Criterion Channel, Shudder, Peacock, Vudu, Tubi, Freevee, Plex

There’s a sizable hole in the monster movie canon: There just aren’t enough good werewolf movies! Ginger Snaps is here to fix that. And if you love Jennifer’s
Body, you’re in for a bloody treat.

Directed by future Orphan Black co-creator John Fawcett, Ginger Snaps is a delightfully gruesome story about two sisters. One of them has gotten her period for
the first time, and is turned into a werewolf shortly after. Things get messy, fast.
</item 10>

<item 11>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
This incredibly dark and squirmy movie is much quieter and more internal than Evil Dead Rise, but it sure puts the screws to the audience, especially as it
builds to an unforgettable, horrifying climax. —TR

Oct. 20: Silent Hill

Where to watch: Peacock

The mid-2000s were a no man’s land for horror movies. By 2006, we were four years out from the blockbuster prestige of Gore Verbinski’s The Ring and the
J-horror import boom that followed it in America, and only just beginning to transition into the Saw-dominated years of “realistic” horror. More importantly, we
were still almost a decade away from the arthouse horror boom that would kick off closer to 2014 with films like It Follows. This was exactly the environment
Silent Hill was released into, and it deserved so much better.

The movie follows Rose, whose daughter, Sharon, is plagued with awful episodes of sleepwalking and nightmares about the mysterious abandoned town of Silent
Hill.
</item 11>

<item 12>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
And a total Five Nights novice might even prefer the sillier, shorter Cage version.
</item 12>

<item 13>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Again, every kid is different, but generally speaking, that 12 or 13 age cutoff is probably about right.

Wasn’t there already a horror movie about evil animatronics?

Yes! Five Nights at Freddy’s took so long to get the movie treatment that it was actually lapped by a spirited knockoff. Willy’s Wonderland doesn’t have nearly
so much lore as the FNAF games, but its premise is basically One Night at Cage’s, with Nicolas Cage playing a man hired to serve as an overnight janitor at a
haunted Chuck E. Cheese-type location. As expected, the interesting element of the movie is Cage’s performance — especially because it’s entirely dialogue-free,
muting one of Cage’s most distinctive features as an actor.

The novelty isn’t quite enough for Willy’s Wonderland to reach the upper echelon of Cage’s last decade, but hardcore fans of either the actor or the concept
might enjoy the strange combination of a unique movie-star brand with an off-brand version of the famous game series.
</item 13>

<item 14>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Can you enjoy Five Nights at Freddy’s if you haven’t played the games?

At a Blumhouse panel for the recent New York Comic Con, producer Jason Blum explained that he thinks a lot of adaptations — books, comics, and games alike —
falter when creators try to simultaneously please hardcore fans and remain accessible to new audiences. According to Blum, his company finally cracked Five
Nights at Freddy’s by deciding not to make it with new audiences in mind. In other words, they ultimately decided to just please the fans.

It was a clear applause-bait line at a convention filled to the brim with, yes, fans. But that doesn’t mean the movie is hard to follow. It starts from scratch,
mimicking the premise of the first game: A guy named Mike (Josh Hutcherson) is hired as overnight security guard for an abandoned Chuck E. Cheese-style arcade
restaurant, and encounters some animatronics with strange and eerie origins. There’s no comics-style backstory that demands pre-movie study.
</item 14>

<item 15>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
(Thomas has plenty of closely held secrets of his own.) And Evans’ commitment to violent mayhem gives this one some memorable moments that are likely to haunt
you later, in the dark. —Tasha Robinson

Oct. 6: Dracula 2000

Where to watch: Max

With the scars of nu-metal having healed and Olivia Rodrigo’s Paramore-infused pop tracks having gone full “vampire,” now is the time to give Gerard Butler’s
half-bare-chest take on Dracula the respect it’s due.

Maybe the most Dimension Films movie to ever bear the Dimension Films studio logo (close second: The Faculty), Dracula 2000 finds a group of thieves — including
of-the-moment faces like Omar Epps, Jennifer Esposito, and Shane West! — transporting a stolen silver coffin from London to New Orleans. They think there’s
treasure inside. Are they idiots? Absolutely. And when they find Dracula instead of gold doubloons or whatever, they unleash hell.
</item 15>

<item 16>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
It’s not always done effectively, but in NYC, Murphy and the writers heighten the foreboding terror of being alone in a city of millions, of feeling helpless in
a crowded room, of knowing that there’s something out to get you but not being able to do anything about that.

The whole season is surprisingly subdued for AHS (yes, even with the sex cages), and it builds up to a painfully poignant finale scene where, for 10 minutes set
to Kraftwerk’s “Radioactivity,” one of the main characters wordlessly trudges through the next 10 years of his life, watching everyone around him slowly succumb
to AIDS, as horror and reality blend together. —Petrana Radulovic

Oct. 14: Cat People

Where to watch: Max

Want to be a smarter horror fan this October? Watch the movie that originated the modern jump scare, and then impress people at Halloween parties with this
knowledge.
</item 16>

<item 17>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Rather than managing to find his family, all he finds is an abandoned mining town. For most of the movie, the women’s frantic running and hiding from monsters
is paralleled by Christopher wandering around the exact same areas, filled with a dreadful sense that his family is close, but completely unable to see them or
the horrors threatening to kill them. It’s not a subtle metaphor for the differences in the ways that men and women go through the world, but it is an effective
one, and it manages to create a level of crushing existential terror and defeat that few movies ever come close to. —AG

Oct. 21: Suspiria (2018)

Where to watch: Prime Video, Freevee

Dario Argento’s 1977 Suspiria is a horror masterpiece, the sort of film even the best directors would face insurmountable odds of matching with a remake.
</item 17>

<item 18>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
The poster child for the misguided term “elevated horror,” and the subject of more than a few memes (particularly around telephone poles), the thing that often
gets lost about Hereditary is that it’s actually really fucking good. And it’s damn scary too.

The movie follows Annie Graham, a difficult mother of two, who just lost her mom. During the funeral service, Annie notices quite a few people are here to mourn
the mother she thought had no friends. She eventually learns this group of old people all belonged to the same bizarre semi-cult her mother did. And that’s
where the witchy stuff starts.

From there everything descends into a complicated mishmash of tightly coiled family drama, supernatural plotting, and years-old resentments, and it’s absolutely
excellent. Who’s to say which is scarier in this movie, the verbal immolation or the literal one?

Even if you’ve seen it already, you probably owe this movie a rewatch. You definitely remember that it’s good, but you probably don’t remember just how great it
really is.
</item 18>

<item 19>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.
</item 19>

<item 20>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
And when they find Dracula instead of gold doubloons or whatever, they unleash hell. Overacted, canted-angled, seven-string-guitar-plucking HELL. Luckily, as
Dracula hunts down local college student Mary Van Helsing (Justine Waddell), with whom he already shares a psychic connection, Mary’s dad’s new vampire-hunter
assistant Simon (young Jonny Lee Miller!) is on his tail.

Butler, never more a heartthrob, plays Dracula in a mesmerizing, baroque manner; think Nicolas Cage doing Dracula if this year’s Renfield was emo
turn-of-the-millennium trash. And he’s constantly biting unsuspecting B-list actors to add to his army of hot people, who fight the heroes in multiple
obligatory scenes of wire-fu. Writer-director Patrick Lussier cuts it all up like it’s the video for “Freak on a Leash,” and, well, objectively, it’s quite
unlike most horror movies you can watch today!
</item 20>

<item 21>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Every person reads like a threat, and every new competitor or friend seems to be looking for a new way to drain the life out of the people around them — until
they finally actually do it. But, like all Refn projects, what really sets The Neon Demon apart is how it looks. The griminess of the movie’s world turns
brightly lit photo shoots into shadow-filled nightmares, with people who stalk the sets like jungle predators.

Though The Neon Demon didn’t get the same acclaim as other so-called elevated horror movies from the time, looking back, it fits perfectly among the ranks of
The Witch, It Follows, and Under the Skin. Refn’s nightmarish modeling movie is dripping with cynicism, cruelty, and a creeping atmosphere, with more meanness
than almost any other movie from the era could muster. But it’s all in the pursuit of beauty, of course. —AG

Oct. 16: Project Wolf Hunting

Where to watch: Hi-Yah!
</item 21>

<item 22>
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
</item 22>

<item 23>
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
</item 23>

<item 24>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Beleaguered movie theaters got a boost this year from film fans who made an event out of seeing two radically different movies back-to-back: a slyly satirical
movie about a living fashion toy, and a heavy, grim historical drama about the father of the atomic bomb. While Barbie and Oppenheimer do have a clear theme in
common, they’re still radically different movies. But the way they collectively dominated the 2023 box office and the cultural discussion is a strong reminder
that regardless of tone or topic, a great story, told passionately, stands out. Cinephiles can be drawn to just about any movie in any genre, if it’s well
crafted and engaging, no matter what kind of fruit it is.

Here at Polygon, we have pretty eclectic collective tastes: Some of us are hungriest for action, horror, epic fantasy, or challenging science fiction, while
some of us prefer complicated drama or dark neo-noir, and others are drawn to musicals, comedy, animated adventures, and other lighter fare.

Our best-of-the-year list reflects that range.
</item 24>

<item 25>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE
</item 25>

<item 26>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Like many folk tales before they’re sterilized by pop culture, Pan’s Labyrinth is a fairy tale full of menace, danger, and horror — but also hope. It is both a
refuge from a turbulent time and an attempt to understand it, a parable that becomes timeless in its specificity with new meaning to discover every time it’s
revisited. Widely considered to be writer-director Guillermo del Toro’s magnum opus, it shows the director operating at the peak of his powers, with lavish
production design, evocative colors, and naturalistic performances juxtaposed with quiet, dangerous magic.

But what makes it a great film is astonishingly simple. Some movies endeavor to make sure you understand them. The best ones help you understand yourself. Pan’s
Labyrinth, in the end, wants to know what kind of person you are by the time the credits roll. —Joshua Rivera

Oct.
</item 26>

<item 27>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—Joshua Rivera

Oct. 23: Eyes Without a Face

Where to watch: Criterion Channel, Max

The late 1950s weren’t exactly a booming era for French horror films. While Henri-Georges Clouzot’s psychological thriller Diabolique received an effusive
reception upon its release in 1955, the prevailing attitude of French critics at the time was that horror was antithetical to the high-minded standard of what
French cinema should aspire to. Eyes Without a Face bucked against this perception, taking what would otherwise be a standard mad scientist tale and turning it
into a dark horror fable about love, guilt, and madness.

The film follows Dr. Génessier, a renowned plastic surgeon wracked with guilt over a car accident that disfigured his beloved daughter Christiane’s face.
</item 27>

<item 28>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
And for the score: Thom Yorke. Yes, Radiohead Thom Yorke.

Where other directors would have obsessively recreated a source text, rebooting it into something unrecognizable, Guadagnino’s film is an homage — the sort of
movie you imagine Argento would make with the budgets and creative freedom afforded by tech companies trying to buy cultural capital. It’s more political. More
grotesque. The ending is just more.

The truth is 2018’s Suspiria and the original Suspiria should be enjoyed together. In the past, that’s been a challenge, with the original film being
surprisingly difficult to watch — especially if you prefer streaming. But this month, Criterion Collection has you covered, allowing for one of the most batshit
double features of your life. —CP

Oct.
</item 28>

<item 29>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
gang. Scooby, Shaggy, Fred, Velma, and Daphne set off in the Mystery Machine to the remote woods of Casper County. The movie is filmed in the same documentary
style that The Blair Witch Project made iconic, and mostly inserts the gang into live-action locations like real woods or a real town where they interview
locals.

The project was originally created as a bumper to bookend segments of a 1999 Scooby-Doo marathon (just a few months after the movie it was based on was
released), but after Cartoon Network saw what the creators put together, it wisely decided to re-air the short segments end-to-end as a movie.

The Scooby-Doo Project not only stands alongside the original Blair Witch as an impressive and loving parody, but also as a similarly unnerving found-footage
horror movie.
</item 29>

<item 30>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—PV

Oct. 11: Heck

Where to watch: YouTube

Skinamarink has proven to be one of the most polarizing horror releases of 2023. Kyle Edward Ball’s feature debut about two children trapped alone inside their
home by a malevolent entity eschews the formal conventions of traditional cinematography and plot, consisting instead of a series of canted-angle shots of dark
hallways and yawning darkness that forces the audience to ruminate on the horrors that lay therein. Personally, I vibed with it heavily, and so it should come
as no surprise that I quite enjoyed Ball’s 2020 short film Heck as well.

Conceived as a “proof of concept” for Skinamarink, the short is told from the perspective of a young child who wakes in the dead of night to the blaring sound
of their mother’s television set. With their mother seemingly nowhere to be found, the child is left to their own devices, with no means of either leaving the
house or calling for help.
</item 30>

<item 31>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—CP

Oct. 22: Pan’s Labyrinth

Where to watch: Available to purchase on Amazon, Apple, and Vudu

Not just a great horror movie, but one of the Great Movies, Pan’s Labyrinth is among the best examples of a fairy tale for adults. Set during the aftermath of
the Spanish Civil War, Pan’s Labyrinth follows Ofelia (Ivana Baquero) after her pregnant mother takes her to live with her new stepfather, Captain Vidal (Sergi
López) — a fascist Falangista hunting down the republican rebels that are hiding out in the Spanish forest.

In this terrifying time, Ofelia encounters a faun, a mythical creature that claims she is the reincarnated Princess Moanna, daughter of the king of the
underworld. The Faun (Doug Jones) tells her that if she completes three tasks, she can return to the king and queen and live forever as princess of the
underworld.
</item 31>

<item 32>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Rose brings Sharon to the town in a last-ditch attempt to find answers, but things go horribly wrong when the fog-shrouded town seems to transport them (along
with police officer Cybil Bennett) to a new monstrous dimension. Meanwhile, after the mother and daughter have been gone for a few hours, Rose’s husband,
Christopher, sets off to Silent Hill to find his family.

Inside the town is where Silent Hill takes the most inspiration from the video game series it’s based on. Some of Silent Hill’s most recognizable enemies and
monsters show up, chasing Cybil, Sharon, and Rose through the town and eventually into the iconic Pyramid Head. These segments are impressively scary, and
steeped in the dreadful atmosphere that made the games famous.

But even beyond the terrifying town itself, the best part of Silent Hill comes when Christopher arrives. Rather than managing to find his family, all he finds
is an abandoned mining town.
</item 32>

<item 33>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
It’s an idea that dates as far back as the spiritualism movement of the late 19th century, and one that takes on a frightening and fascinating dimension in
Kiyoshi Kurosawa’s post-Y2K horror-thriller Pulse.

Alternating between two storylines, Kurosawa’s film follows a group of young adults and university students as a rash of inexplicable disappearances and
suicides occurs around Tokyo. As the protagonists investigate further, they slowly uncover a shocking revelation: The souls of the dead are spilling over into
the mortal world and ensnaring their unsuspecting prey through the very infrastructure of the internet. By the time they realize this, however, it’s too late;
the unearthly contagion has taken on a life of its own, and the only way to survive is cling to what few connections they have left.
</item 33>

<item 34>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Typical enough for an adaptation, some of the more complicated lore has been stripped out, though it could probably be reintroduced in any sequel. This does
make the movie more accessible for franchise newcomers, whether Blum wanted that or not.

Also, fans might expect more screen time than they actually get for the core animatronics team of Freddy, Bonnie, Chica, Foxy, and Carl the Cupcake.

Is Five Nights at Freddy’s safe for kids?

For an ostensible horror franchise — one based on nostalgia for and subversion of animatronic characters that date to the 1980s — the Five Nights at Freddy’s
games have a surprising amount of young fans. Its kid appeal is an echo of a time when playgrounds were alight with elementary school kids describing the antics
of Freddy and/or Jason, whether witnessed firsthand or pieced together from rumors.
</item 34>

<item 35>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The Texas Chainsaw Massacre 2 is nothing like that. You won’t see its name in a film school syllabus, and the films it inspired can be counted on one hand.
Nonetheless, there’s something commendable about the gall of Tobe Hooper, the original Texas Chain Saw director, returning to his series with the energy of a
middle-aged man on his first ayahuasca retreat.

Leatherface gets in a high-speed car battle, eventually standing on the roof of a speeding sedan. Dennis Hopper dual-wields chainsaws. A radio station office is
obliterated like it’s a novelty rage room. Sometimes you want a steak, and sometimes you want a hamburger. Texas Chainsaw Massacre 2 is a damn juicy burger. —CP

Oct. 27: Possessor

Where to watch: Hulu

My first Brandon Cronenberg (yes, David’s son) movie was Infinity Pool, his latest messed-up meditation on the rich and powerful.
</item 35>

<item 36>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Our best-of-the-year list reflects that range.

How the Polygon top 50 list works

Every year, the staff’s film fans create individual ballots to reflect their top movies, with an option to rank them numerically or just weight them. We use
these ballots to generate a collective list, weighted by strength of opinion to make sure a much-seen mainstream movie doesn’t have too much advantage over a
terrific but underseen indie. And then we put it all out there for your enjoyment. The top 10 picks on each our staff’s lists will be listed in the comments.

Any movie released in the US this calendar year is eligible, but since we are publishing this in early December, some December releases are underrepresented. We
hope you’ll find a new favorite here on our list of the best movies of 2023.

Honorable mentions Movies that received votes but did not crack our top 50: The Pope’s Exorcist, Bottoms, Rye Lane, No Hard Feelings, Hundreds of Beavers, Evil
Dead Rise, Guardians of the Galaxy Vol.
</item 36>

<item 37>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
There’s no comics-style backstory that demands pre-movie study.

Still, complete newcomers checking out the movie to see what the fuss is about may be confused for other reasons. The movie’s backstory is parceled out
awkwardly, in ways that are sometimes confusing. Characters’ decisions don’t always make sense. And there’s as much family melodrama as thrill-ride scares.
There isn’t much you can do about it except assure yourself that any confusion you experience in watching it isn’t just you. The movie is kind of weird, and not
because of some missing fan-only information that unlocks all its inconsistencies. The for-the-fans vibe Blum described applies more to whether the average
person will care about this story, not whether they can understand it.

Will Five Nights at Freddy’s experts enjoy the movie more?

Longtime fans hungering for an exploration or expansion of the game series’ continuing mythology should prepare themselves to see something more simplified.
</item 37>

<item 38>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.
</item 38>

<item 39>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
I vividly remember telling friends how exciting and debauched the movie was, and their response surprised me: “Clearly you haven’t seen Possessor.”

That night, I rectified this, and dear readers: They were right. As disturbing and mystifying as I found the world of Infinity Pool, Possessor was all that and
more. And I haven’t been able to get it out of my brain since. Which is… fitting, considering the movie.

Andrea Riseborough (Mandy, To Leslie) plays an elite corporate assassin who is able to inhabit other people’s bodies, getting closer to her victims by posing as
people already in their lives. It’s a haunting premise, and one Possessor takes full advantage of, both in the moments of action and in the assassin’s
contemplation about what her identity truly even is.

Possessor is an unforgettable entry into the psychological horror canon, and it brings a little sci-fi into your Halloween celebrations. Don’t miss it. —PV

Oct.
</item 39>

<item 40>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
This is why it made our list of the best Blu-rays of 2023.

But reader, while I’m one of those sickos who obsesses over video resolution and bitrate, I’ll be honest: You don’t need the fanciest version of Creepshow to
enjoy its comedic chills. In fact, the most aesthetically harmonious format might be a poorly treated VHS tape that recorded the film off TBS in the mid-1990s.

Horror anthologies have most recently been used to condense as much terror into a tiny package, like chugging a shot glass of hot sauce. Creepshow is a
throwback to a softer, more leisurely style of horror — a middle ground between the plodding tinglers of Poe and contemporary slashers.

Plus, Stephen King performs a one-person show as a hillbilly. So like, that should be enough. —Chris Plante

Oct. 4: Pulse

Where to watch: Prime Video

What if technology could be used to communicate with the dead?
</item 40>

<item 41>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Cat People is not only a key part of horror movie history, it’s a delightfully dark and sexy time positively radiating with atmosphere. The movie follows a
woman (Simone Simon) who is caught between her desire for a new man in her life (Kent Smith) and her belief that she is cursed to turn into a panther if she
becomes aroused. And at 73 minutes, it’s one of the shorter horror classics you can watch this fall.

Now for that jump scare origin. There’s a tense sequence in this movie where one character stalks another. Director Jacques Tourneur lets the tension build to
an unbearable level before surprising the audience with the sudden appearance of a bus. Often considered the first example of the modern jump scare, it was
called the “Lewton Bus” after producer Val Lewton, who used the technique again in later movies. —PV

Oct.
</item 41>

<item 42>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
From there, things get really strange.

Shot on 35mm, A Wounded Fawn is a gorgeous movie to behold, even before the action truly starts. Once it does, be prepared for a descent into the
phantasmagoric, with jaw-dropping visuals, eerie practical effects, and a heavy dose of Aeschylus’ Eumenides.

With fantastic performances from Sarah Lind (Jakob’s Wife) and Dropout veteran Josh Ruben (Werewolves Within) in an unsettlingly off-type role, A Wounded Fawn
is a bold new entry into the serial killer movie canon. —PV

Oct. 19: The Hole in the Ground

Where to watch: Max

Few horror tropes get as much mileage as the image of the unnatural child. Likely that’s because the idea that kids should be innocent and cheerful is baked
into so many human cultures that a child acting spooky and inhuman is a terrific cross-cultural recipe for horror. But an effective creepy-kid movie requires a
spectacular creepy kid performer.
</item 42>

<item 43>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
In the presence of imminent horror, our first instinct is to scramble for answers and, in the absence of any to be found, our second instinct is to search for a
means of escape. Through a captivating lead performance and deft sound design, Egan delivers a gripping horror story in just under 10 minutes. —TE

Oct. 26: The Texas Chainsaw Massacre 2

Where to watch: Max

The Texas Chain Saw Massacre is a nauseating smoothie of rural horror, outré indie experimentation, and wannabe snuff film. Its murders aren’t real, but you’d
be forgiven for being convinced the film’s actors are in genuine trouble; from the budget film stock to the filthy sets, it all feels… off. The result is not
just a great horror flick, but a monumental work that originated many tropes of its genre and, most recently, earned a coveted spot in the Sight and Sound list
of the greatest films of all time.

The Texas Chainsaw Massacre 2 is nothing like that.
</item 43>

<item 44>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Produced after Kurosawa’s Cure (which we featured on last year’s Halloween calendar), Pulse is widely celebrated as one of the foundational texts in the canon
of J-horror cinema, alongside Hideo Nakata’s Ring and Takashi Shimizu’s Ju-on: The Curse. Eerie and methodical, Kurosawa’s film is a nihilistic meditation on
technology and human relationships that presages an advent of loneliness in a world growing more and more “connected” with each passing day. Rife with imagery
that will stay with you long after it’s over, Pulse is a tremendous and terrifying movie worth experiencing. —TE

Oct. 5: Apostle

Where to watch: Netflix

Don’t confuse Gareth Edwards (director of The Creator and Rogue One: A Star Wars Story) with Gareth Evans, director of the Raid movies. If you do, you may sit
down to the Netflix-exclusive Apostle expecting expansive, detail-driven science fiction, and be unpleasantly surprised when you get a grimy, extremely gory
period piece that goes to startling extremes.
</item 44>

<item 45>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

Whether it’s something gory and macabre, silly and irreverent, or eerie and unsettling, the genre of horror is as rich and varied as the multitude of ghosts,
ghoulies, and homicidal maniacs that go bump in the night.

Looking for the best horror films available to stream on Netflix, Hulu, Max, and Paramount Plus? No worries, we’ve got the goods. We’ve combed through the
libraries of each of the major streaming platforms to bring you a list of our most recommended horror movies. Here are the best horror movies you can stream
right now, from old classics to new hits. Our latest update added Encounters of the Spooky Kind.
</item 45>

<item 46>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
This is not a horror movie whose horrors rely on brutality and gore — though there is certainly that and then some — but one whose apocalyptic premise provokes
both outright terror and disquieting introspective dread. —TE

18. Mission: Impossible — Dead Reckoning Part One

Director: Christopher McQuarrie

Cast: Tom Cruise, Rebecca Ferguson, Hayley Atwell

Where to watch: Digital rental/purchase

Tom Cruise loves popcorn and movies, and nobody made a better popcorn movie in 2023 than Dead Reckoning Part One. While it fell some way short of the series’
critical and commercial peak, Fallout — mistakes were made, chiefly putting “Part One” in the title — the seventh (seventh!) Mission: Impossible film is another
shockingly confident and artful action movie that delivers set-piece after astounding set-piece over a breathless two and a half hours.
</item 46>

<item 47>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Things get messy, fast.

Ginger Snaps doesn’t just do a great job filling in the werewolf canon. It’s a grisly body horror story placed onto an effective puberty allegory, with a
compelling central sister relationship. If you’ve ever watched something like Sam Raimi’s Spider-Man and thought, This is great and surprisingly gooey, but give
me stories like this about teen girls, then Ginger Snaps is for you.

The movie has amassed a dedicated cult following in the years since its release. You’re next up to join the ranks — see you at the next full moon! —Pete Volk

Oct. 3: Creepshow

Where to watch: Kanopy

The new Creepshow home video release — presented in vibrant 4K UHD Blu-ray — is the best way to revisit this horror anthology classic. You can see individual
bubbles in the foam of the ocean as it slowly drowns Ted Danson. This is why it made our list of the best Blu-rays of 2023.
</item 47>

<item 48>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
The film was adapted from John Ajvide Lindqvist’s 2004 novel of the same name, which inspired not just this Swedish film, but a 2010 American adaptation, a
comic-book prequel, and two stage plays. The latter has its own legacy — it was adapted by the magnificent National Theater of Scotland, and it eventually had a
run at St. Ann’s Warehouse in 2015. Few books inspire so much additional great art. So I suppose I’m recommending the book just as much as the film. —Chris
Plante

Malignant

Director: James Wan

Cast: Annabelle Wallis, Maddie Hasson, George Young

Where to watch: Max

There was just no way to see it coming. After the Conjuring and Insidious franchises, plus blockbuster turns with Furious 7 and Aquaman, James Wan could have
cashed in chips to make another moody franchise-starter to stretch his jump-scare muscles.
</item 48>

<item 49>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
As the interminable night drags on, with hours morphing into days morphing into weeks morphing into months of penumbral isolation, the child grows more fearful
and feral, resorting to increasingly more desperate acts of disobedience in hopes of rousing their mother from sleep. This effort, however, is ultimately proven
to be in vain.

Heck circles the same rough ideas and themes of Skinamarink to different effect, channeling the vulnerability of a child and the horror of abandonment and
neglect to create an experimental horror experience that demands the audience’s full attention. It’s a fascinating companion to Skinamarink that shows just how
far Ball has come as a director honing in on this particular strain of horror, and makes it all the more intriguing to speculate on what he might conjure up
next. —TE

Oct. 12: Harper’s Island

Where to watch: Available to purchase on Amazon, Apple, and Vudu

In theory, slasher TV should be easier to pull off.
</item 49>

<item 50>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

If there’s one thing we love here at Polygon, it’s the Halloween season.

We cover horror year-round, whether it’s the best horror movies you can stream at home or the best horror movies on Netflix, and generally, we have our finger
on the undead pulse of the newest and ghastliest releases in horror. We even have a list of the best horror movies of the year (ranked by scariness, of course).

Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.
</item 50>
=== END ITEM id=2607fd8c69e096d1 ===

=== ITEM id=6cd4dba487b001b4 ===
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 2>

<item 3>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 3>

<item 4>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
showed that Epic knew it was breaking bad at the time it did the deed.

But Epic mentioned Project Liberty in its own opening statement — so, by that point, it had already been an hour since it admitted it intentionally broke
Google’s rules. “Epic decided to stand up because that’s what you do to a bully,” Bornstein told the jury.

“All we know is whatever is in the destroyed chats, as bad as the documents are, is worse.”

And it’s possible no examination will be able to take the stink off one of Google’s ugliest moves: the one where Google employees up to and including CEO Sundar
Pichai were caught setting sensitive chats to auto-delete to keep them out of a court’s hands. The court has already decided Google should be sanctioned in some
way for making potential evidence disappear, and Bornstein used it to plant persistent seeds of doubt in the minds of jury members. “All we know is whatever is
in the destroyed chats, as bad as the documents are, is worse.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 5>

<item 6>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 7>

<item 8>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 10>

<item 11>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 11>

<item 12>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 12>

<item 13>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
That was our proposal.

Oh, I think I read that one in court. I didn’t realize that was the same thing. It’s so far off from what I heard [about a “special billing exception”] that I
didn’t recognize it being that. Does winning the verdict in this Google case help your appeal with the Apple case in any way?

There’s no linkage between the cases and law, so it would just come down to whether the court is in any way following current events on this topic. But there’s
no legal connection between the two. It really comes down to the decisions of the humans involved: the justices and the support teams involved when considering
the appeal.

Google will appeal this; they’ve told us they’re going to challenge the verdict. Will Epic rely on this ruling in the meantime?

We’re not going to wait. We’re going to do absolutely everything we can as quickly as we can to start changing the world. We not only have this verdict here in
the United States, it is a worldwide verdict, right?
</item 13>

<item 14>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 14>

<item 15>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 15>

<item 16>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So this trial has been four years in the making. I look back at a September 2019 email that was in discovery about a plan to draw Google into a legal battle
over antitrust. Can you describe what it felt like, after four years, to hear the jury find for Epic?

Well, it’s a great day for all developers to see that the Sherman Antitrust Act works in the new era of tech monopolies; we’ve not had a major antitrust verdict
against a tech company that meant change and benefits for everybody since the 1990s, with the US v. Microsoft. Back in the early days of the internet. So this
is an awesome thing and it’s much needed by the industry which is being strangled by a few gatekeepers imposing insane amounts of control and extracting huge
taxes, which not only raise prices for consumers but also make a lot of kinds of products just unviable.

I understand the significance, but... you were there in person.
</item 16>

<item 17>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 17>

<item 18>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Who will Epic and Google call as witnesses?

We are going straight to the top of the org chart, folks. Google CEO Sundar Pichai and Epic CEO Tim Sweeney are absolutely expected to attend. Android
co-founder Andy Rubin might be deposed, and Google says we’ll hear from current Android (and Chrome, and Chrome OS) boss Hiroshi Lockheimer. Google also tells
us representatives from Apple, Netflix, Motorola, and AT&T should testify as well.

How much hypocrisy can we expect in the courtroom?

I’d say quite a bit! For example:

Epic says it’s not in this for the money, it’s only suing to end Google’s monopoly — but Epic would surely profit if it doesn’t have to pay Google’s cut.

It looks like Epic’s going to argue that you’re locked into Android after buying an Android phone, but I can personally attest that switching back and forth
between the platforms is easier than ever.
</item 18>

<item 19>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 19>

<item 20>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 20>

<item 21>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 22>

<item 23>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 23>

<item 24>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 24>

<item 25>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Enough that the judge has decided to explicitly tell the jury we’re not seeing all the evidence, because Google employees all the way up to CEO Sundar Pichai
were caught setting those conversations to auto-delete. This issue has come up in the US v. Google antitrust trial in Washington, DC — still ongoing at the
start of Epic’s trial — as well.

And the dirt isn’t all on Google’s side. Epic was caught straight-up admitting it premeditated the whole plan to bypass Google’s payment systems. Since Google
is countersuing over that, it could be a big deal.

When you say “straight-up admitting...”

Here is an email from Epic CEO Tim Sweeney that, I think, speaks for itself:

“Here is our official plan for communicating with Google about bypassing the Google Play Store: SAY NOTHING TILL IT SHIPS” Image via Epic v. Apple

We’ll have a whole cast of intriguing people called as witnesses during the trial who might dish even more.

Who will Epic and Google call as witnesses?
</item 25>

<item 26>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 26>

<item 27>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 27>

<item 28>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 28>

<item 29>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Also, that one’s not over till the Supreme Court weighs in or declines to take a look.

2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!

3) Oh, and by the way, this is a jury trial.

Wait, why does that matter?

Epic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)

Maybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping
security holes will sway a jury against Epic instead? Who knows!

(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)

Didn’t other parties sue Google too?

They did! And then, they all settled.
</item 29>

<item 30>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 30>
=== END ITEM id=6cd4dba487b001b4 ===

=== ITEM id=893ad400bbf57ad4 ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement also meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement also meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: The party described uses its own store.
  Fact 2.1: The party described uses its own payments.
  Fact 2.2: The party described forces developers to all have the same terms.
  Fact 2.3: The party described forces OEMs to all have the same terms.
  Fact 2.4: The party described forces carriers to all have the same terms.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 1>

<item 2>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
It’s not like how the watch uses its EKG capabilities to detect atrial fibrillation. On consumer wearables, SpO2 features generally have no medical use cases.
They’re essentially for “entertainment” purposes, and we’ve repeatedly reported that you should never, ever use them as any sort of replacement for traditional
pulse oximeters. The only real blow for Apple is if the rumors are correct and it’s working on sleep apnea detection. Losing the sensors in that instance would
make developing this feature harder. However, that feature would require a lengthy FDA clearance process. That means more advanced health features like sleep
apnea detection are more of a medium to long-term concern — and it’s very possible that Apple will have a workaround by the time such features are ready for
consumers.

That said, for this route to be successful, Iancu says it depends on the specific patent, ITC ruling, and what exactly Apple has planned as far as redesigning
the product.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
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
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 5>

<item 6>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
Listen to TC+ editor Alex Wilhelm and I talk more about how there’s plenty of market share to go around in Friday’s episode of the Equity podcast below. — Mary
Ann

PayPal is being sued for alleged “Draconian” payment policies

On October 5, Mary Ann broke the news that PayPal has been hit with a class action lawsuit by consumers represented by law firm Hagens Berman alleging that the
fintech giant’s anti-steering rules stifle competition against lower-cost payment platforms such as Stripe and Shopify. Specifically, according to an
investigation conducted by the firm’s consumer rights attorneys, PayPal has subjected consumers to excess charges when purchasing from online merchants that
accept PayPal or Venmo.

Someone commented on social media that they didn’t see the problem with what PayPal is doing and questioned whether or not Visa and Mastercard do the same
thing. So I went back to the attorneys who filed the lawsuit, who noted that the “anti-steering rules” are not the first of their kind.
</item 6>

<item 7>
The Apple Watch’s double tap gesture points at a new way to use wearables (The Verge, 2023-10-25T17:04:01+00:00)
That lack of choice is also apparent in third-party apps. My email app lets me double-tap to start a quick reply using Siri, but even in 2023, I am not
dictating my emails. Apps like Spotify and Pocket Casts don’t work with double tap, either. Most third-party apps that do work with double tap just let you
dismiss notifications. That’s helpful but still limiting.

With watchOS 10, apps like Fitness and Weather have been redesigned to be more glanceable. Instead of one long screen that you scroll endlessly through,
information is divided into more digestible chunks. You still scroll, but there’s less of it. This is great, except I often want to use double tap to navigate
through menus instead of the digital crown. And I can’t.

You can scroll through the new widget Smart Stack, but if I want to select a widget and open it, I still have to use my other hand.
</item 7>

<item 8>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 10>

<item 11>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
They said: “Visa and MasterCard once imposed similar anti-steering rules on merchants accepting their cards but, after the Justice Department sued the networks
for antitrust violations, they agreed in 2010 to eliminate their anti-steering rules as part of the settlement. With payments transitioning into the digital
realm, PayPal has simply ripped a page right from the Visa and MasterCard [sic] playbook.”

Meanwhile, Patrick McGahan, a partner at Scott+Scott who focuses on antitrust litigation, had an interesting take that actually involves both of those card
giants. He told TechCrunch that the case illustrates that “the tensions between merchants and the payment system providers are not over, and that litigation
regarding this key cost faced by merchants is likely to continue for some time.”

He added: “Platform companies that operate as gatekeepers in their market, such as PayPal, will continue to be the subject of antitrust litigation as a result
of the significant fees they charge.
</item 11>

<item 12>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 12>

<item 13>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 13>

<item 14>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 15>

<item 16>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
“From a legal perspective, whatever change Apple makes has to be significant enough to avoid the patent, and it cannot be what’s called a ‘colorable’
difference. Minor changes will not be sufficient,” Iancu says.

Basically, Apple can’t simply make a small tweak to the code and call it a day. Masimo claims that the patents are hardware related, so even completely
disabling the sensor may not work depending on how the patent is written. Apple could remove it entirely from the design, but that would require mucking around
with Apple’s meticulous supply chain and could take some time.

Then there are the options that would cost Apple some money. Apple could sit down with Masimo and hammer out a licensing deal for the infringed patents. In a
New York Times interview, Masimo CEO Joe Kiani said Apple “had not engaged in licensing negotiations.” It’s an easy-seeming fix, so why not do it? For starters,
both parties have to want to settle.
</item 16>

<item 17>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Where does Apple go from here?

A hail mary veto is unlikely, but that doesn’t mean Apple is just going to accept an import ban on a $17 billion segment of its business. Apple spokesperson
Nikki Rothberg told The Verge in a statement that the company was “pursuing a range of legal and technical options to ensure that Apple Watch is available to
customers.” That means the watch is going back on sale one way or another — it’s just a question of what path Apple takes.

Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months. The company will also probably ask for a
stay on the import ban, but Brittingham says that’s not likely to be granted because Apple would have to prove that it would be irreparably harmed. In other
words, it’d have to make a convincing case that it would go out of business if it can’t sell the Apple Watch.

Apple says it’s exploring both legal and technical options to get around this import ban.
</item 17>

<item 18>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.
</item 18>

<item 19>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Like Masimo, AliveCor argued that Apple stole its tech to develop the wearable EKG feature, thereby infringing on its patents. However, while the two cases look
similar, the result of one doesn’t speak to how the other will play out.

“Every case is different, every patent is different,” says Levi. “Different complainants who bring up a case have got their own businesses, their own products,
and those implicate a whole different set of facts.”

The Apple Watch SE will remain unaffected because it doesn’t have a blood oxygen sensor. Photo by Amelia Holowaty Krales / The Verge

Case in point, while the ITC granted AliveCor’s import ban, it hasn’t had any impact because the Patent Trial and Appeal Board ruled that AliveCor’s EKG tech
isn’t actually patentable. AliveCor would have to win its appeal to that ruling before any Apple Watches with EKG features would be pulled from the market for
that reason.
</item 19>

<item 20>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
But in this case, the important thing is that the ITC sided with Masimo. In January 2023, it ruled that Apple Watches did in fact infringe on Masimo patents.

The ITC then issued an import ban in October, as well as an order to stop selling products infringing on Masimo’s patents that had already been imported. We’re
currently nearing the end of a 60-day presidential review period, in which President Joe Biden or the US Trade Representative (USTR) has the opportunity to veto
the ban. If there’s no veto by the time the review period ends, the ban will go into effect.

But will Biden actually veto?

Apple getting a presidential veto would be like lightning striking the same place twice.
</item 20>

<item 21>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 22>

<item 23>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
For starters, both parties have to want to settle.

“Apple is a tough nut to crack,” says Brittingham, noting that Apple is one of the most sued companies in Silicon Valley. The company’s large cash reserves also
make taking Apple to court a costly endeavor. Masimo, for example, has reportedly spent $60 million litigating the issue thus far. “I think they tend to hold
out as long as possible in order to make it less appetizing to sue Apple.”

The other option is for Apple to avoid the import ban altogether. Iancu says Apple could simply decide to manufacture the watches within the US instead of
overseas — but that’s a significant supply chain change that’s unlikely to happen any time soon.

Does this set a precedent?

This import ban isn’t the only one Apple is facing. In February, the Biden administration also declined to veto another ITC import ban and cease-and-desist
order stemming from a dispute with AliveCor over the Apple Watch’s EKG features.
</item 23>

<item 24>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Soon after, Apple reportedly poached several Masimo engineers and its chief medical officer. And then in fall 2020, Apple released the Apple Watch Series 6 —
its first Apple Watch to feature an SpO2 sensor to measure blood oxygen saturation levels.

The SpO2 sensors measure blood oxygen levels and are at the heart of this patent dispute. Photo by Amelia Holowaty Krales / The Verge

In 2020, Masimo filed a lawsuit in the US District Court in the Central District of California accusing Apple of stealing trade secrets and infringing on 10 of
its patents. That case dragged on and on, so Masimo then filed a separate case with the ITC in 2021. Then last year, Apple sued Masimo back, claiming the
company made an Apple Watch clone with its Masimo W1 Medical Watch. Clearly, there’s some bad blood here. But in this case, the important thing is that the ITC
sided with Masimo.
</item 24>

<item 25>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 25>

<item 26>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Apple getting a presidential veto would be like lightning striking the same place twice.

“It is extraordinarily rare for the President to overturn a decision from the International Trade Commission and in my view, [it’s] unlikely in this particular
case,” says Andrei Iancu, co-chair of the Council for Innovation Promotion (C4IP) and partner at Sullivan & Cromwell LLP, who was also the director of the US
Patent and Trademark Office from 2018 to 2021.

To get a veto, a company usually has to show a public interest or health policy basis — and that isn’t the case here. “It’s unlikely that the import exclusion
order will be disapproved by the USTR and the president,” says Smith Brittingham, partner at Finnegan, Henderson, Farabow, Garrett & Dunner, LLP. “The only real
answer would be, ‘Well, it’s Apple.’”

That said, Apple did manage to beat the system once. In 2013, then President Barack Obama vetoed an import ban on the iPhone.
</item 26>

<item 27>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 27>

<item 28>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Earlier this year, the US International Trade Commission (ITC) ruled that Apple had infringed on two patents from medical device maker Masimo. As a result, the
ITC said it would impose an import ban on the Apple Watch Series 9 and Ultra 2 starting December 26th. At the time it was hard to believe that this would
actually come to pass: Apple gets sued all the time, and even when it loses, how often does it actually face dramatic consequences?

Well, now would be one of those times. Experts say that, barring a Christmas miracle, it’s unlikely that Apple will find a way to escape the ban. Case in point,
the company shocked everyone yesterday when it decided to preemptively pull the watches from its online store starting December 21st at 3PM ET. And after the
24th, they’ll disappear from Apple Stores, too.
</item 28>

<item 29>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 29>

<item 30>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?
</item 30>
=== END ITEM id=893ad400bbf57ad4 ===
