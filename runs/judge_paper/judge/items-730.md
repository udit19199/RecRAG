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

This file has 2 items.

=== ITEM id=411622eb47f732d2 ===
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
Thankfully, their helpful hitchhiker companion Baby knows just the place they can get help: the Firefly mansion, where her family lives. From there, things
devolve into a host of horrific stage plays, cursed experiments, and gruesome murders.

The family at its center, whom Rob Zombie went on to make two more movies about, is repulsive and absolutely horrible, but still zany enough to be truly
unsettling in a way that few horror movies can be. House of 1000 Corpses is a grisly, grotesque film that captures all the grime of the ’70s movies that
inspired it. But what really makes it worth watching is the way it balances silliness and terror, doling each out in proportional quantities until you can’t
figure out if you should laugh or scream. Instead, you just end up doing both. —AG

Oct. 30: Thirst

Where to watch: Peacock

Park Chan-wook has never missed.
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
—AG

Oct. 16: Project Wolf Hunting

Where to watch: Hi-Yah!; available to rent on Amazon, Apple, and Vudu

Sometimes, what you’re looking for is just literal tons of blood and gore. And when that’s contained in a premise that’s essentially “Con Air meets Predator,
with a bit of The Last Voyage of the Demeter,” that’s hard to beat.

Project Wolf Hunting is a Korean monster mash built out of an extremely fun idea. “What if a bunch of dangerous criminals rebelled on the cargo ship they were
being transported on, only to discover the ship is also transporting a monster?”

The story is necessarily light, and the characters are thinly drawn (although the effective costuming and makeup work go a long way), but that’s not what you’re
here for if you’re watching Project Wolf Hunting. You’re here for gory violence, and boy, is there a lot of it.
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
Rather than managing to find his family, all he finds is an abandoned mining town. For most of the movie, the women’s frantic running and hiding from monsters
is paralleled by Christopher wandering around the exact same areas, filled with a dreadful sense that his family is close, but completely unable to see them or
the horrors threatening to kill them. It’s not a subtle metaphor for the differences in the ways that men and women go through the world, but it is an effective
one, and it manages to create a level of crushing existential terror and defeat that few movies ever come close to. —AG

Oct. 21: Suspiria (2018)

Where to watch: Prime Video, Freevee

Dario Argento’s 1977 Suspiria is a horror masterpiece, the sort of film even the best directors would face insurmountable odds of matching with a remake.
</item 15>

<item 16>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Rose brings Sharon to the town in a last-ditch attempt to find answers, but things go horribly wrong when the fog-shrouded town seems to transport them (along
with police officer Cybil Bennett) to a new monstrous dimension. Meanwhile, after the mother and daughter have been gone for a few hours, Rose’s husband,
Christopher, sets off to Silent Hill to find his family.

Inside the town is where Silent Hill takes the most inspiration from the video game series it’s based on. Some of Silent Hill’s most recognizable enemies and
monsters show up, chasing Cybil, Sharon, and Rose through the town and eventually into the iconic Pyramid Head. These segments are impressively scary, and
steeped in the dreadful atmosphere that made the games famous.

But even beyond the terrifying town itself, the best part of Silent Hill comes when Christopher arrives. Rather than managing to find his family, all he finds
is an abandoned mining town.
</item 16>

<item 17>
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
It’s not always done effectively, but in NYC, Murphy and the writers heighten the foreboding terror of being alone in a city of millions, of feeling helpless in
a crowded room, of knowing that there’s something out to get you but not being able to do anything about that.

The whole season is surprisingly subdued for AHS (yes, even with the sex cages), and it builds up to a painfully poignant finale scene where, for 10 minutes set
to Kraftwerk’s “Radioactivity,” one of the main characters wordlessly trudges through the next 10 years of his life, watching everyone around him slowly succumb
to AIDS, as horror and reality blend together. —Petrana Radulovic

Oct. 14: Cat People

Where to watch: Max

Want to be a smarter horror fan this October? Watch the movie that originated the modern jump scare, and then impress people at Halloween parties with this
knowledge.
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
=== END ITEM id=411622eb47f732d2 ===

=== ITEM id=6ec3588ca93dcc0d ===
Gold sentences and facts:
Sentence 0: After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week 9 as a big part of the offense with Josh Dobbs under center.
  Fact 0.0: Hockenson has had three limited practices this week.
  Fact 0.1: The writer will trust that Hockenson is ready to go.
  Fact 0.2: Hockenson continued to play through the ailment in Week 9.
  Fact 0.3: Hockenson was a big part of the offense in Week 9.
  Fact 0.4: Josh Dobbs was under center for the offense when Hockenson played in Week 9.
Sentence 1: Kincaid was a limited participant in practice on Wednesday due to a shoulder injury, but it’s been reported he’s likely to suit up on Saturday at less than 100%.
  Fact 1.0: Kincaid was a limited participant in practice on Wednesday.
  Fact 1.1: Kincaid's limited practice participation was due to a shoulder injury.
  Fact 1.2: It has been reported that Kincaid is likely to suit up on Saturday.
  Fact 1.3: It has been reported that Kincaid is likely to play on Saturday at less than 100%.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Perhaps it's all health related and Thomas' volume from last week will stick.

ZONE COVERAGE: Atlanta usually plays a lot of zone coverage early on and if it works they stick with it for the whole game (Week 5 was an example of that). In
Weeks 3 and 4 against the Lions and Jaguars they played much more man coverage, especially when they needed a fourth-quarter turnover. And because Howell is
much less explosive against zone looks, I expect the Falcons to stick with zone much more this week.

Start Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs.
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 2>

<item 3>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
This week might not matter for many fantasy football owners. Some have secured their spot in the postseason; others are well out of contention. However, for
those in must-win situations, this week really matters. It's not the time to have a hole in your lineup at tight end, but if you do, we're here to help with our
Week 14 fantasy TE PPR rankings.

Fortunately, there are several solid sleepers and streaming options despite two tough byes and some big injuries to notable TEs. It's always tough to know who
to trust at this volatile position, so we break down which tight ends have higher floors for those favored to win comfortably and which have higher ceilings for
those who need to really shoot for big points.

Let's get to it and check out the TE landscape for Week 14.

Who are the TEs on bye in Week 14?
</item 3>

<item 4>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The fact that he has 13 total targets in his past three games combined makes it tough to buy into him suddenly getting 10-plus targets this week. The matchup is
also a problem -- the Browns have the defensive backs to hang tight with a slower-than-normal Kupp in man-to-man coverage, which they play in a lot. The Browns
also have the pass rush to force Matthew Stafford to get the ball out quickly and without much distance. I'm not that nervous to bench Kupp for Rashee Rice,
Courtland Sutton and both of the top Seattle WRs.
</item 4>

<item 5>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Who are the TEs on bye in Week 14?

Few would've expected Week 14's two-team bye to be as problematic as it is, but losing Logan Thomas and Trey McBride this late in the season could have some
fantasy owners scrambling.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | D/STs | Kickers

Fortunately, both the Commanders and Cardinals have above-average TE defenses, so taking those off the board only helps the position this week.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy TEs in Week 14?

As usual, Travis Kelce (vs. Bills in Week 14) tops our rankings, but T.J. Hockenson (@ Raiders), Sam LaPorta (@ Bears), David Njoku (vs. Jaguars), and Dalton
Kincaid (@ Chiefs) aren't far behind.
</item 5>

<item 6>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 6>

<item 7>
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
</item 7>

<item 8>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Packers) always has a high ceiling as the Giants' de facto No. 1 receiver.

Tyler Higbee (neck) was banged up last week, and he's no sure thing to suit up this week against Baltimore. Even if he does, he's not a recommended play against
the Ravens' top-10 TE defense.

We'll be updating these TE PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy TE PPR Rankings Week 14: Who to start, sit at tight end

Rankings based on full-point PPR formats

Rank Player

1 Travis Kelce, Chiefs vs. Bills

2 T.J. Hockenson, Vikings @ Raiders

3 Sam LaPorta, Lions @ Bears

4 David Njoku, Browns vs. Jaguars

5 Dalton Kincaid, Bills @ Chiefs

6 George Kittle, 49ers vs. Seahawks

7 Isaiah Likely, Ravens vs.
</item 8>

<item 9>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 9>

<item 10>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST

Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST SITS: Kyle Pitts, Tyler Conklin, Tyler Allgeier, both QBs

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Pittsburgh's biggest change in offensive scheme last week involved Kenny Pickett throwing over the middle much more than before (Pat Freiermuth benefitted). But
they also ran the ball effectively at Cincinnati -- Najee Harris had one of his best games this year. Expect more of the same against the Cardinals this week.
Arizona has allowed five touchdowns to tight ends over its past five games but actually has been relatively efficient defending them otherwise.
</item 10>

<item 11>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 11>

<item 12>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Again, you can't expect more than WR3 production, which is where his overall scoring has been in 2023.

Start: WR Tyler Lockett

While Metcalf is WR34 in terms of total points this season in half-point PPR, Lockett is only two notches higher at WR32. They finished as WR13 and WR18,
respectively, last season, so it's a big dropoff. Lockett also has a decent matchup vs. Stephon Gilmore. He, like Metcalf, remains a volume-based WR3.

WEEK 13 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: WR Jaxon Smith-Njigba

Smith-Njigba can be a bit interesting if you are desperate in a deeper league because he has the matchup on par with the other two wideouts vs. Jourdan Lewis.
</item 12>

<item 13>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 13>

<item 14>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 14>

<item 15>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Both Darrell Henderson and Royce Freeman (@ Cardinals) would be decent dart throws in a week filled with byes, but their floors in a zero-bye week seem too low
to trust them if Kyren Williams is back and at 100 percent. The matchup is juicy, but we have no idea what to expect from a touch-share perspective, so these
guys are no better than desperation, TD-dependent flexes.

Speaking of guys coming back from injury, we saw Khalil Herbert (@ Vikings) return to the field for Chicago against the Lions last week. However, his
performance left a lot to be desired. He finished with just 35 yards on 16 carries and wound up vastly overshadowed by Justin Field's 100-yard rushing effort.
Even worse, D'Onta Foreman scored the Chi's lone rushing TD on the afternoon.

I don't care what your team looks like — you shouldn't be starting Chuba Hubbard or Miles Sanders (@ Titans).
</item 15>

<item 16>
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
</item 16>

<item 17>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 17>

<item 18>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ CHI MIN -2.5 O/U 44.5 OPP VS WR 20th PROJ PTS 10.7 WR RNK 33rd YTD Stats REC 13 TAR 26 REYDS 166 TD 2 FPTS/G 8.3 No doubt there's
some potential here, but some hunches keep me from falling for Osborn this week. One: I think the Vikings will try to get their run game going since they don't
have Justin Jefferson. Two: Jordan Addison is far more explosive and T.J. Hockenson has been a much more relied-upon weapon. They should have more targets than
Osborn. And the expected zone coverage Chicago will play further will limit Osborn unless he sees more targets than expected.
</item 18>

<item 19>
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
</item 19>

<item 20>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 20>

<item 21>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 21>

<item 22>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
If they're trailing, JSN should see action with a negative game script.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

TE Brevin Jordan, Texans @ Jets ($3,100). Jordan was a great streaming option and fill-in for the crowd that lost Dalton Schultz (hamstring) on Sunday, hauling
in three-of-four targets for 64 yards against the Broncos. If Schultz is out again, Jordan is a serviceable option that has touchdown upside against a reeling
Jets team that is busy booking vacations or plotting how to find themselves on a different roster in 2024. If Schultz does play, we'll pivot to the Jets' Tyler
Conklin, who is $100 less and has plenty of opportunity in an even better matchup.

FLEX WR Jameson Williams, Lions @ Bears ($3,500).
</item 22>

<item 23>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 23>

<item 24>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He's still worth trotting out there over Logan Thomas, Dallas Goedert and Cole Kmet, but keep expectations around 70 yards on six catches.

LAST WEEK: Came alive against the Texans' zone-heavy defense with season-highs in targets (11), catches (seven) and yards (87). This was his second game of his
last three where his target share was over 23%. It was nice to see him as a key part of the offense, but I can't help but notice he played a season-low 53% of
the snaps.

FILM: Two things helped Pitts out: One, he looked a little quicker off the snap and a little faster when running straight. He's still lumbering when he changes
directions in his routes but perhaps he's slowly getting healthier. Two, Desmond Ridder seemed very comfortable throwing into tighter windows to Pitts in zone
coverage as the two had some good timing down, mainly on dig and out-breaking routes.
</item 24>

<item 25>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
For those in deeper leagues, Tyler Conklin (vs. Texans), and Cade Otton (@ Falcons) both have matchups against defenses that entered last week in the top five
for FPPG allowed to TEs. Brevin Jordan (@ Jets) and Tanner Hudson (vs. Colts) also have highly favorable matchups and can work as deep-league PPR plays, though
Jordan's outlook is entirely dependent on whether Dalton Schultz (hamstring) is healthy.

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy TE busts in Week 14?

Dallas Goedert (@ Cowboys) isn't a "must-start" if he returns from his forearm injury this week. That said, he's also not a "must-sit." He had three catches and
50 yards against Dallas in Week 9 before exiting early with the forearm injury, so he can clearly produce against this defense.
</item 25>

<item 26>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But they float around league average in catch rate allowed (77.8%, below league average) and yards per catch (8.9, better than league average). However, no team
has allowed more first-down receptions to tight ends (17). Keep in mind, this defense hasn't seen tight ends better than Evan Engram, Sam LaPorta or Dalton
Schultz.

COMMANDERS: Throw to their tight ends on 26.3% of their throws, fourth-most in football. But that hasn't always meant a huge target share for Thomas. In Weeks 1
and 5 he saw at least 22.9% of Sam Howell's targets. He left Week 2 after suffering a concussion and was absent in Week 3; then was not part of the game plan
much in Week 4 (7.7% target share against Philadelphia). Perhaps it's all health related and Thomas' volume from last week will stick.
</item 26>

<item 27>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 27>

<item 28>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Terrell on Collins and make things a little harder on the Texans. Terrell shadowed a receiver for the first time this season in Week 4, contributing to Calvin
Ridley's modest day. It ultimately would mean multiple bodies on Dell.

Sit Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 7.6 TE RNK 13th YTD Stats REC 11 TAR 21 REYDS 121 TD 0
FPTS/G 5.8 If Coach Smith is right, then holding on to Pitts in Fantasy makes sense. But it's hard to see the good in his game so far this season beyond him
finding holes in zone coverage.
</item 28>

<item 29>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 29>

<item 30>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 30>

<item 31>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Brown is also questionable but did not practice Friday due to a knee injury.

Panthers at Buccaneers (-5.5)

Carolina may be down two tight ends this weekend as the club has ruled out Hurst and is listing Tremble as doubtful. This is the third straight game Hurst will
miss. The team will also be without Bell and will likely look to rookie Alex Cook to start. Meanwhile, corner CJ Henderson (concussion) was a full participant
on Friday and carries no designation heading into the weekend.

Tampa Bay is pretty banged up coming into Week 13, particularly on the defense side of the ball. The Bucs will be without key pieces David, Dean, and White when
they play host to Carolina. Meanwhile, Godwin did not practice Friday after being added to the injury report on Thursday with a neck injury.

49ers (-3) at Eagles

Each of the 49ers' questionable players were limited Friday and did not practice the previous two days.
</item 31>

<item 32>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Jourdan Lewis. JSN can mirror some of the production Curtis Samuel gave the Commanders last week in Dallas, but forget 100 yards. Around 50 yards seems like the
ceiling with a big boost from garbage time.

Sit: TEs Noah Fant and Will Dissly

Noah Gray is a better fantasy player than Noah Fant this season. Will Dissly is barely a better fantasy player than Will Mallory this season.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: Seahawks D/ST

The Seahawks are getting better defensively against the pass, but that's not enough to trust them here, as they're getting gashed on the ground, too.

Sit: K Jason Myers

This is actually not a good spot for one of the most reliable fantasy kickers, given the Cowboys don't give up many field-goal attempts.
</item 32>

<item 33>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 33>

<item 34>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

What are the biggest QB injuries in Week 14?

Leading into Week 13, Dorian Thompson-Robinson (concussion) was the lone QB injury situation we were closely monitoring. Even if Thompson-Robinson is able to
clear concussion protocol in time for Week 14, we expect Flacco to make his second start as a Brown.

This week, we unfortunately saw Kenny Pickett (ankle) and Derek Carr (head, shoulder, back) exit their respective contests with injuries, complicating their
statuses for Week 14. We've replaced both QBs with backups Mitchell Trubisky and Jameis Winston, respectively, as early indications point toward both missing
time.

UPDATE: Trevor Lawrence suffered an ankle injury on Monday, which will likely keep him out for at least a week and open the door for C.J. Beathard (vs. Browns)
to start.
</item 34>

<item 35>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 35>

<item 36>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 36>

<item 37>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
The hope is we keep seeing that for Smith, especially if the game is competitive like most Falcons games tend to be. He certainly hasn't done anything to lose
his role. I don't mind chancing it with Smith as a streaming tight end, though I just don't have the conviction to start him over Pitts.

LAST WEEK: Was the second time in three weeks where Smith outproduced Pitts, but the first time it was so lopsided. The difference in snaps played and routes
run is marginal -- Pitts did more of both -- but Smith is taking advantage of shorter routes with a quarterback who doesn't like to challenge downfield all that
much. What's more, Smith averaged more yards per catch not just in Week 4 but also in Week 2, and he was behind Pitts by less than a yard in Week 3 when both
had at least eight targets.

FILM ON PITTS: There's no change in his explosiveness whatsoever.
</item 37>

<item 38>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
This is the good advice that you should take: Be wary of anyone who's rushed for just 47 yards two straight weeks despite getting significant duty with Kenneth
Walker III (oblique) on the shelf. Charbonnet's also not doing much in the passing game, and this is as bad of a matchup as San Francisco was last week. You
should do better, even in a big bye.

WEEK 13 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Start: WR DK Metcalf

Metcalf can flash big plays and stay involved in games...until he disappears when Smith can't get the ball to him. Smith might be judicious about throwing his
way with second-year corner sensation DaRon Bland in coverage. That said, you simply can't sit DeKaylin Zacharius against anyone because of his big-game
ability.
</item 38>

<item 39>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Bears (DK: $4,700 | FD: $6,200 | SD: 1.3x)

Njoku went nuclear last weekend, catching six of his eight targets from Joe Flacco for 91 yards and two touchdowns. Now he gets a chance for an encore against
the Bears, who have surrendered the 10th-most PPR points to tight ends this season. No joke, dude — you can trust Njoku!

More solid Week 15 TE values: Logan Thomas, Commanders at Rams (DK: $3,700 | FD: $5,100 | SD: 1.5x); Cade Otton, Buccaneers at Packers (DK: $3,200 | FD: $5,200
| SD: 1.65x); Tucker Kraft, Packers vs.
</item 39>

<item 40>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Jonnu even has more red-zone targets than Pitts (by one, three to two). The matchup could create some receiving opportunities, so maybe he catches five passes
for 50 yards, but that's not anything to be excited about. I'd rather start Jake Ferguson, Zach Ertz and Dallas Goedert.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Jonnu Smith TE ATL Atlanta • #81

Age: 28 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 5.2 TE RNK 24th YTD Stats REC 15 TAR 20 REYDS 179 TD 0
FPTS/G 8.2 I feel like Smith is doing what Pitts is supposed to do: get open and make plays after the catch.
</item 40>

<item 41>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 41>

<item 42>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 42>

<item 43>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
It's been a journey back. He's going to get there."

FILM ON SMITH: From what I see, Smith is running shorter routes and is getting open faster than Pitts. That's earned him some targets from Ridder, who seemed to
have better timing with Smith than Pitts. Smith is not drawing any kind of tight coverage like Pitts either, mostly because he's staying closer to the line of
scrimmage. But after the catch Smith has been a handful, breaking three tackles just last week and picking up extra yards with it. He's not fast, but he's
faster than Pitts, and for at least last week and Week 2 it changed the calculus for how the Falcons divvied up targets.

TEXANS: Have had a bundle of troubles with tight ends, giving up an 84% catch rate and 6.76 YAC/reception to the position, both bottom-six in the league.
</item 43>

<item 44>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Commanders clearly value the position in their offense, Thomas sees the most throws from Howell, and if Atlanta plays heavy zone coverage then it should
mean a bunch of underneath throws for Thomas to take in. You're hoping for at least seven targets here, which is possible. He's worth the nod over Jake
Ferguson, Dalton Schultz and Tyler Conklin, and it's close in full PPR over Kyle Pitts.

LAST WEEK: Thomas was a big-time benefactor of the Commanders chasing points against Chicago's heavy zone coverage, seeing 10 of his 11 targets once the Bears
built a 17-point lead in the first half. A 24-yard catch on the final play of the first half against prevent defense didn't hurt either.

FALCONS: See the third-most targets per game to tight ends (9.0) which in turn has made them bottom-five in raw stats like total catches and yards allowed.
</item 44>

<item 45>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 45>

<item 46>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Seahawks

7 Isaiah Likely, Ravens vs. Rams

8 Cole Kmet, Bears vs. Lions

9 Jake Ferguson, Cowboys vs. Eagles

10 Dallas Goedert, Eagles @ Cowboys

11 Kyle Pitts, Falcons vs. Buccaneers

12 Gerald Everett, Chargers vs. Broncos

13 Evan Engram, Jaguars @ Browns

14 Brevin Jordan, Texans @ Jets

15 Tyler Conklin, Jets vs. Texans

16 Cade Otton, Buccaneers @ Falcons

17 Tanner Hudson, Bengals vs. Colts

18 Pat Freiermuth, Steelers vs. Patriots

19 Tucker Kraft, Packers @ Giants

20 Chigoziem Okonkwo, Titans @ Dolphins

21 Dawson Knox, Bills @ Chiefs

22 Jonnu Smith, Falcons vs. Buccaneers

23 Juwan Johnson, Saints vs. Panthers

24 Michael Mayer, Raiders vs.
</item 46>

<item 47>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
As Watson eases his way back in, it's not like Doubs will fall off a cliff in terms of production. He remains relevant with that type of connection forming.

FLEX TE Sam LaPorta, Lions ($10,000). Rookie tight ends aren't typically fantasy-relevant right away, but that's where LaPorta finds himself, becoming a
favorite target of Goff. He's been on an upward trajectory, starting with five targets in Week 1, six in Week 2, and 11 in Week 3. He's totaled 39, 63, and 84
yards, respectively, in those games, scoring a touchdown in the most recent contest. The Packers are a middle-of-the-road group against tight ends, but they
still haven't faced one in a passing offense that lives up to LaPorta's level. On a short week with some banged-up players on offense, LaPorta should be a
heavily involved security blanket.
</item 47>

<item 48>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'm pretty certain they're not bad enough to make Tim Boyle look like Tim Tebow or even Tim Hasselbeck, but I do wonder if they're juuust bad enough to give
better-than-normal numbers to Garrett Wilson and Breece Hall. Reinforcements along the O-line should help the Jets' cause against a Falcons defense that's
blitzed at a league-average rate, is eighth-lowest in pass rush pressure rate and is fifth-lowest in sacks (28).
</item 48>

<item 49>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 49>

<item 50>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs WAS ATL -2.5 O/U 42.5 OPP VS TE 10th PROJ PTS 10.3 TE RNK 9th YTD Stats REC 18 TAR 32 REYDS 208 TD 0 FPTS/G 7.8 If last week had
been Week 1 then I would have been encouraged by Pitts' performance and would probably keep expectations high. But we've seen four really measly weeks from
Pitts before Week 5, so forgive me for not rushing back to him as a must-start tight end. The truth is that the matchup is a little tougher for him and for
Ridder, plus teams have beaten the Commanders on deep shots to wide receivers, not lumbering tight ends, so I'm not convinced Pitts will maintain his target
share or develop more explosive plays.
</item 50>
=== END ITEM id=6ec3588ca93dcc0d ===
