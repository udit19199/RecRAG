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

=== ITEM id=8748d4aafb87a2f8 ===
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
You’re here for gory violence, and boy, is there a lot of it. The director has bragged that the team used 2.5 tons of fake blood while filming the movie, and it
certainly shows. It’s one of the goriest movies I’ve ever seen, and that gore is augmented by terrific sound design, which builds the monster up into something
well beyond the realm of humankind by making its movements sound metallic. It’s a machine designed for killing, and Project Wolf Hunting is a movie designed for
a bloody good time. —PV

Oct. 17: Kuroneko

Where to watch: Criterion Channel, Max

Like his 1964 film Onibaba (which we featured in our 2021 Halloween countdown), Kaneto Shindo’s 1968 supernatural horror film is a cerebral, gorgeous, and eerie
drama set amid the tumultuous Sengoku period of Japanese history.
</item 1>

<item 2>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.
</item 2>

<item 3>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
You don’t get the full gale force of this film, its pleasures and its overarching point, without Scorsese reaching back to his past work, and relying on viewers
to recognize his voice in the storytelling.

Killers of the Flower Moon

Killers of the Flower Moon is perhaps most closely related to The Aviator as a film that appears to be a recognizable, industry-friendly form of awards-season
prestige. It’s also a return to Scorsese’s core themes and characters. It’s another cast of avarice-fueled, violent white American dumbasses gaining ill-gotten
wealth and making the country incrementally worse. It’s again about those men failing up, consolidating power, and shading the country more evil.

Killers adapts a major literary work of historical true crime from a New Yorker writer. The book delves into an obscure but monstrous historical incident that
is a cosmic truth in miniature, an evil that speaks to the heart of the entire dark American project.
</item 3>

<item 4>
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
</item 4>

<item 5>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
—Petrana Radulovic

45. The Origin of Evil

Director: Sébastien Marnier

Cast: Laure Calamy, Suzanne Clément, Dominique Blanc

Where to watch: Digital rental/purchase

This French thriller is the kind of twisty story that’s hard to sum up in any way without giving something away, but that’s part of the appeal. Sébastien
Marnier, director of the shocking 2018 prep school thriller School’s Out, steers The Origin of Evil away from a neo-noir tone, even though the plot is pure
noir, right down to the character types. When a woman (Laure Calamy) insinuates herself into a rich family, claiming she’s a lost daughter to the patriarch,
Serge (Jacques Weber), viewers will suspect she has more of an agenda than she’s letting on. But Serge and the rest of his caustic family members have agendas
too.
</item 5>

<item 6>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
On screen, it’s Scorsese’s first true Western, filled with gorgeous vistas and Oklahoma horizons that would bring a tear to the eye of John Ford, one of
Scorsese’s many idols.

But the film is much stranger than it sounds on paper. Scorsese’s take on the material largely operates outside the text, centering the story on a peripheral
character we know little about from David Grann’s book. Miraculously, the film also has its funny elements, incorporating The Departed and The Wolf of Wall
Street’s nervy energy, featuring long and zany improvisational exchanges on grim subject matter, sudden wild swings in tone, and one major decision that should
spark more debate and differences of opinion than any narrative device Scorsese has ever employed. It’s a suspenseful, moving film with many brilliant parts.

Killers is a classic Scorsese Catholic reckoning.
</item 6>

<item 7>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

Making a collaborative list of the year’s best movies usually feels like trying to form a committee to definitively rank apples and oranges. How do you account
for personal tastes? For access, when a given movie — er, fruit variety — is only available in certain markets at certain times? For the fact that apples and
oranges are just so qualitatively different that they don’t have a lot in common except juiciness and vitamin C content?

Fortunately, 2023 has its own peculiar answer, in the form of the Barbenheimer phenomenon.
</item 7>

<item 8>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
gang. Scooby, Shaggy, Fred, Velma, and Daphne set off in the Mystery Machine to the remote woods of Casper County. The movie is filmed in the same documentary
style that The Blair Witch Project made iconic, and mostly inserts the gang into live-action locations like real woods or a real town where they interview
locals.

The project was originally created as a bumper to bookend segments of a 1999 Scooby-Doo marathon (just a few months after the movie it was based on was
released), but after Cartoon Network saw what the creators put together, it wisely decided to re-air the short segments end-to-end as a movie.

The Scooby-Doo Project not only stands alongside the original Blair Witch as an impressive and loving parody, but also as a similarly unnerving found-footage
horror movie.
</item 8>

<item 9>
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
</item 9>

<item 10>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
2016’s Silence is in conversation with, if not a synthesis of, the ideas explored in The Last Temptation of Christ and Kundun; all three are searching,
anti-commercial religious films Scorsese struggled for years to get made.

In all Scorsese’s late-period films (along with the passel of rock documentaries he’s made during this era, including his two on Bob Dylan and one on George
Harrison), Scorsese is consciously returning to certain pet subjects and periods in his work — not to repeat himself, but to revise and add greater perspective.
There’s a sense that he’s trying to adjust his positions with the clear eye of an older person who has seen more of the world, dropped his pretensions, and lost
his capacity for bullshit. His later films return to his old themes and obsessions, but in the process, they’ve become more contemplative.

New tricks

This era has included some radical departures for Scorsese as well.
</item 10>

<item 11>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Cat People is not only a key part of horror movie history, it’s a delightfully dark and sexy time positively radiating with atmosphere. The movie follows a
woman (Simone Simon) who is caught between her desire for a new man in her life (Kent Smith) and her belief that she is cursed to turn into a panther if she
becomes aroused. And at 73 minutes, it’s one of the shorter horror classics you can watch this fall.

Now for that jump scare origin. There’s a tense sequence in this movie where one character stalks another. Director Jacques Tourneur lets the tension build to
an unbearable level before surprising the audience with the sudden appearance of a bus. Often considered the first example of the modern jump scare, it was
called the “Lewton Bus” after producer Val Lewton, who used the technique again in later movies. —PV

Oct.
</item 11>

<item 12>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
— Austen Goslin

Climax

Director: Gaspar Noé

Cast: Sofia Boutella, Romain Guillermic, Souheila Yacoub

Where to watch: Showtime

Climax isn’t for the faint-of-heart — and we’re saying that in the context of a horror movies list. The movie is set at an all-night dance party inside a
gymnasium, which turns sour after someone spikes the sangria with a little too much LSD. Climax is told in beautifully disorienting long takes that go from
dozens of minutes of uninterrupted and propulsive dance sequences to hazy walks through hallways as the camera mimics the dizzy stumbling of the movie’s
characters. As the psychedelics kick in, so too do some of the attendees’ long-held feuds, leading to disastrous and horrifying consequences. It’s rare that a
movie truly defies description, but if you’ve got a strong stomach and a will to see something you haven’t before, Climax is the perfect movie for you.
</item 12>

<item 13>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.
</item 13>

<item 14>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
If you do know Żuławski’s name, it’s likely for his 1981 psychological horror film Possession, a film whose cult status among horror connoisseurs has only been
amplified in the decades since its release by its difficulty to obtain on physical media or to view online. Fortunately for everyone, that’s no longer the case.

Set in Cold War-era West Berlin, Żuławski’s film stars Jurassic Park’s Sam Neill as Mark, a Russian spy who returns home to find that his wife, Anna (Isabelle
Adjani), has left him and wants a divorce. When Anna refuses to divulge why, only saying that she has not left him for someone else, Mark grows suspicious and
has her tailed. What he eventually discovers is a horrifying secret beyond his comprehension, one which awakens a long-dormant wellspring of anxiety,
resentment, and despair between the two that threatens to tear apart not only their small family, but their very sanity as well.
</item 14>

<item 15>
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
</item 15>

<item 16>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Viggo Mortensen stars as Saul Tenser, a world-renowned performance artist who, alongside his partner Caprice (Léa Seydoux), stands on the cutting edge — both
literally and figuratively — of this cultural phenomenon. When Saul’s activities catch the attention of a mysterious group of evolutionary activists, as well as
the lascivious eye of a government employee named Timlin (Kristen Stewart), he’s forced to confront what he — and everyone else around him — is changing into,
and whether what that is can even be considered “human” anymore.

As macabre as it is moving, grotesque as it is sensuous; Crimes of the Future is an exquisite work of science fiction horror where surgery is the new sex and
our very bodies have rebelled against us for the incalculable destruction we have inflicted on the planet. It’s a film that exists in intimate conversation with
the anxieties of our present, as well as one that represents a stunning return to form for one of cinema’s most forward-thinking directors.
</item 16>

<item 17>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Like many folk tales before they’re sterilized by pop culture, Pan’s Labyrinth is a fairy tale full of menace, danger, and horror — but also hope. It is both a
refuge from a turbulent time and an attempt to understand it, a parable that becomes timeless in its specificity with new meaning to discover every time it’s
revisited. Widely considered to be writer-director Guillermo del Toro’s magnum opus, it shows the director operating at the peak of his powers, with lavish
production design, evocative colors, and naturalistic performances juxtaposed with quiet, dangerous magic.

But what makes it a great film is astonishingly simple. Some movies endeavor to make sure you understand them. The best ones help you understand yourself. Pan’s
Labyrinth, in the end, wants to know what kind of person you are by the time the credits roll. —Joshua Rivera

Oct.
</item 17>

<item 18>
This horror flick boasts the best opening 13 minutes of a film you’ll see this year (The Age, 2023-11-15T18:30:00+00:00)
Save Log in , register or subscribe to save articles for later. Save articles for later Add articles to your saved list and come back to them any time. Got it
Normal text size Larger text size Very large text size It was 16 years ago that horror filmmaker Eli Roth first introduced the world to his slasher vision
Thanksgiving (tagline: “There will be no leftovers”), inspired by the seasonal horror films of his ’70s and ’80s youth such as Halloween (1978), My Bloody
Valentine (1981) and Silent Night, Deadly Night (1984). That was as a spoof movie trailer in the middle of Grindhouse, Quentin Tarantino and Robert Rodriguez’s
pulpy 2007 double-feature, where an axe-wielding pilgrim wreaked havoc on lusty teenagers around the backwoods of Plymouth, Massachusetts.
</item 18>

<item 19>
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
</item 19>

<item 20>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Don’t miss it. —PV

Oct. 28: Opera

Where to watch: Shudder, Tubi, Vudu

You can’t get more than five minutes into a conversation about Italian horror without mentioning Dario Argento, the giallo master.

Opera represents the last hurrah of Argento’s hot streak throughout the ’70s and ’80s. It’s an extravagant horror-thriller that combines high-minded aesthetics
with the viscerality of slasher cinema. The film follows Betty, the understudy to a famous opera singer, who is selected to play the role of Lady Macbeth after
the star suffers a terrible accident. While preparing for her first big performance, Betty is subjected to a terrifying ordeal when she is attacked by a
mysterious unseen assailant who tapes needles under her eyes and forces her to watch as her boyfriend is mutilated. Barely escaping with her life, Betty seeks
the aid of Marco, the play’s director, in apprehending the culprit before they can claim another victim.
</item 20>

<item 21>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
If you love the Wolfenstein games or Call of Duty’s Zombies mode, or B-movies with A-movie production, this one is for you. —PV

Oct. 25: Curve

Where to watch: YouTube

Tim Egan’s 2016 horror short inspires the same visceral physical response as watching an episode of Squid Game. A young woman wakes to discover her twisted body
perched precariously on the slope of a strange cliff overlooking a bottomless pit. Her hands caked with blood, the woman attempts to inch herself away from the
pit, each desperate motion teetering on the cusp of survival and oblivion. Will she be able to escape where so many others have failed?

Curve works as a horror short because it leaves so much to the imagination. We don’t know who this woman is, how she ended up in this situation, or what the
purpose of this place even is.
</item 21>

<item 22>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
At a tight 13 episodes, Harper’s Island knows what it’s about (killing people) and gets right to it (brutally). And yet, the beauty of Harper’s Island and its
antics is how long it’s able to sustain a levelheaded approach to a purely absurd And Then There Were None scenario, right down to how long nobody knows they’re
being picked off. Almost no death is like the last, which seems purely impractical from a serial killing perspective but makes for great TV. The episode titles
are the onomatopoeias for how people die, for Christ’s sake! It’s all very fun and twisty, as bodies and mystery continue to mount. Harper’s Island makes
slasher TV look fun and easy. More TV should be like Harper’s Island. —Zosha Millman

Oct. 13: American Horror Story: NYC

Where to watch: Hulu

I have already gone on record about how I cannot look away from the trash fire that is most seasons of American Horror Story.
</item 22>

<item 23>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—Joshua Rivera

Oct. 23: Eyes Without a Face

Where to watch: Criterion Channel, Max

The late 1950s weren’t exactly a booming era for French horror films. While Henri-Georges Clouzot’s psychological thriller Diabolique received an effusive
reception upon its release in 1955, the prevailing attitude of French critics at the time was that horror was antithetical to the high-minded standard of what
French cinema should aspire to. Eyes Without a Face bucked against this perception, taking what would otherwise be a standard mad scientist tale and turning it
into a dark horror fable about love, guilt, and madness.

The film follows Dr. Génessier, a renowned plastic surgeon wracked with guilt over a car accident that disfigured his beloved daughter Christiane’s face.
</item 23>

<item 24>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
And a total Five Nights novice might even prefer the sillier, shorter Cage version.
</item 24>

<item 25>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
—PV

Possession

Director: Andrzej Żuławski

Cast: Isabelle Adjani, Sam Neill, Heinz Bennent

Where to watch: Shudder, AMC Plus, Metrograph

Outside of the most ardent of cinephile circles, Andrzej Żuławski isn’t a name that inspires enthusiastic recognition in the United States. Known for his
transgressive brand of arthouse cinema, Żuławski’s career was stymied by Communist authorities in his homeland of Poland, with many of his early films being
either heavily censored, banned, or, in one instance, nearly destroyed upon release. It also doesn’t help that the few films of his that have been released in
the States have since gone out of print — though that appears to be changing soon.
</item 25>

<item 26>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Can you enjoy Five Nights at Freddy’s if you haven’t played the games?

At a Blumhouse panel for the recent New York Comic Con, producer Jason Blum explained that he thinks a lot of adaptations — books, comics, and games alike —
falter when creators try to simultaneously please hardcore fans and remain accessible to new audiences. According to Blum, his company finally cracked Five
Nights at Freddy’s by deciding not to make it with new audiences in mind. In other words, they ultimately decided to just please the fans.

It was a clear applause-bait line at a convention filled to the brim with, yes, fans. But that doesn’t mean the movie is hard to follow. It starts from scratch,
mimicking the premise of the first game: A guy named Mike (Josh Hutcherson) is hired as overnight security guard for an abandoned Chuck E. Cheese-style arcade
restaurant, and encounters some animatronics with strange and eerie origins. There’s no comics-style backstory that demands pre-movie study.
</item 26>

<item 27>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE
</item 27>

<item 28>
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
</item 28>

<item 29>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
It’s an absolute miracle Scorsese and Schoonmaker manage to produce a masterpiece out of the chaotic pieces of this production process (again in post, with
ample voice-over). Everyone is letting loose, and to hear the actors and the director describe the shoot after the fact, it takes on something of a religious
quality, with Scorsese as the wrinkled shaman handing out the peyote.

What grounds it is the framing. Scorsese is telling us a familiar story we’ve come to expect from him, about criminals and the pleasure they take in committing
crimes and reaping the benefits. The brilliance is the way he mines filmgoers’ long relationship with his work, repurposing the style and tone of his gangster
classics and inserting it into the world of finance. As he’s done effectively since Casino, Scorsese weaponizes postmodernism, using viewers’ presumed existing
familiarity with his films as context.
</item 29>

<item 30>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Happy Friday, Polygon readers! Each week, we round up the most notable releases to streaming and video rental, highlighting the biggest and best new movies for
you to watch at home.

This week’s biggest debut is Spider-Man: Across the Spider-Verse, which is now streaming on Netflix. That’s not all, as Insidious: The Red Door — the fifth
installment in the Insidious horror franchise — also arrives on the platform this week alongside Jawan, one of the biggest Indian action movies of the year.
There’s plenty more exciting releases this week too, with A Haunting in Venice now streaming on Hulu, the Italian superhero movie Freaks vs. the Reich on Prime
Video, plus the premiere of The Kill Room and Outlaw Johnny Black from director-star Michael Jai White on VOD.

Here’s everything new to watch this weekend!
</item 30>

<item 31>
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
</item 31>

<item 32>
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
</item 32>

<item 33>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
Exorcism is a specific form of prayer that the [Catholic] Church uses against the power of the devil."

The Catholic Church distinguishes between two forms of exorcisms, said Reehil. Exorcisms are either "major" or "minor."

"A major exorcism is performed only for one, who is demonstrably possessed by a demon or demons, and may only be performed by a priest the bishop has
specifically permitted to do so," said Reehil.

Conversely, a minor exorcism is associated with sacraments and blessings, said Reehil.

"Demonologists distinguish three levels of demonic incursion," explained Reehil. These three levels are temptation, obsession (which is also called
"oppression") and possession.
</item 33>

<item 34>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
The Killer is the kind of movie that can throw in a fight scene for five minutes that’s better than almost any fight in a mainstream movie this year, then never
go back to hand-to-hand combat again. In other words, it’s a David Fincher movie through and through.

Fincher is famous for his precision on set and his dedication for finely honing every aspect of his films until they’re exactly what he needs. And yet, his
movies never feel stodgy, tight, or lifeless; he makes it look easy. And that’s where The Killer truly excels. It breezes by every assassin movie and thriller
trope, executing each one flawlessly, all while making a snide comment about it. It’s the kind of movie that feels so effortless and smooth that it makes you
wonder why all movies can’t be this good. But that’s Fincher’s gift; his films have the kind of quality and handmade perfection so precise it can be mistaken
for the work of factory machines… until you look a little closer. —AG

16.
</item 34>

<item 35>
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
</item 35>

<item 36>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Again, every kid is different, but generally speaking, that 12 or 13 age cutoff is probably about right.

Wasn’t there already a horror movie about evil animatronics?

Yes! Five Nights at Freddy’s took so long to get the movie treatment that it was actually lapped by a spirited knockoff. Willy’s Wonderland doesn’t have nearly
so much lore as the FNAF games, but its premise is basically One Night at Cage’s, with Nicolas Cage playing a man hired to serve as an overnight janitor at a
haunted Chuck E. Cheese-type location. As expected, the interesting element of the movie is Cage’s performance — especially because it’s entirely dialogue-free,
muting one of Cage’s most distinctive features as an actor.

The novelty isn’t quite enough for Willy’s Wonderland to reach the upper echelon of Cage’s last decade, but hardcore fans of either the actor or the concept
might enjoy the strange combination of a unique movie-star brand with an off-brand version of the famous game series.
</item 36>

<item 37>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Rudolf is careful to never speak about his job while at home.

This may sound like it sidelines the tragedy and horror of the Holocaust, centering the story on the culprits rather than the victims, but Glazer’s carefully
measured detachment lets the situation speak for itself, as the quiet part gets louder and more horrific throughout the film. This creates a completely
different kind of Holocaust film from almost any we’ve ever seen, and one that is essential in understanding the scale and depth of its evils. —AG

21. Priscilla

Director: Sofia Coppola

Cast: Cailee Spaeny, Jacob Elordi, Dagmara Domińczyk

Where to watch: Theaters

I won’t spoil it, but perhaps the best needle drop of the year arrives at the end of this Priscilla Presley biopic: a stone-cold tear-jerker so apt it takes
your breath away, and a song with a complex history that adds multiple layers of delicious dramatic irony to the scene. Nobody ever accused Sofia Coppola of
lacking taste.
</item 37>

<item 38>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Instead, he made Malignant, a high-emotion giallo stuffed into dingy ’90s direct-to-video pastiche like some kind of horror-movie turducken. Wan pulls back the
layers in an almost tedious fashion: The pregnant Madison (Annabelle Wallis) is first the victim of domestic abuse, then she encounters another killer, and then
she starts dealing with psychotic episodes tied to her childhood imaginary friend Gabriel, and theeeeen it’s revealed… Well, please go behold it.

Strung together with a melodramatic cover of The Pixies’ “Where Is My Mind,” reveling in horror tropes to the point of parody, the final twists of Malignant are
some of the most gratifying lunacy of the year, and the acrobatic actor Marina Mazepa brings it all home in a display of gruesome ballet. I won’t explain
anything more out of fear of spoilers — just get on the Malignant train. Wan put his dream (nightmare?) on screen for us all to enjoy.
</item 38>

<item 39>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The Texas Chainsaw Massacre 2 is nothing like that. You won’t see its name in a film school syllabus, and the films it inspired can be counted on one hand.
Nonetheless, there’s something commendable about the gall of Tobe Hooper, the original Texas Chain Saw director, returning to his series with the energy of a
middle-aged man on his first ayahuasca retreat.

Leatherface gets in a high-speed car battle, eventually standing on the roof of a speeding sedan. Dennis Hopper dual-wields chainsaws. A radio station office is
obliterated like it’s a novelty rage room. Sometimes you want a steak, and sometimes you want a hamburger. Texas Chainsaw Massacre 2 is a damn juicy burger. —CP

Oct. 27: Possessor

Where to watch: Hulu

My first Brandon Cronenberg (yes, David’s son) movie was Infinity Pool, his latest messed-up meditation on the rich and powerful.
</item 39>

<item 40>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
— AG

In the Mouth of Madness

Director: John Carpenter

Cast: Sam Neill, Julie Carmen, Jurgen Prochnow

Where to watch: YouTube

Among the wildest movies John Carpenter has ever made (and that’s saying something), In the Mouth of Madness follows insurance investigator John Trent (Sam
Neill), who is hired to find a missing fame horror novelist. Things become increasingly unhinged as the plots of the author’s books and the various monsters
seem to invade the real world. Neill, a staple of this list, is absolutely fantastic responding to the horrors of hell, slowly becoming exactly as off-kilter as
they are. By the time the movie makes it to the third act, the door to hell is halfway open and Trent is ready to dive headfirst into the void, which is
honestly how every movie’s third act should go.

This is also the third in Carpenter’s apocalypse trilogy, which also includes two other stone-cold classics, The Thing and Prince of Darkness. They aren’t on
this list, but you should watch them anyway.
</item 40>

<item 41>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Yep — along with everything else, Poor Things is also goth Barbie.

Adapted from a cult novel by the eccentric Scottish writer, illustrator, and typesetter Alasdair Gray, Poor Things reworks Frankenstein to explore themes of
feminism, sex, and social rot. Lanthimos doubles down on the baroque stylings of The Favourite to create an astonishingly visually dense film composed of
outlandish costumes, lavish sets, and fantastical painted backgrounds, often shot through woozy fish-eye lenses. It’s also, surprisingly, one of the year’s most
hilarious movies, in no small part thanks to Mark Ruffalo’s game campery and McNamara’s cleverly twisted syntax. But the movie is Stone’s. Her performance is
fearless, frank, funny, and intensely physical.
</item 41>

<item 42>
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
</item 42>

<item 43>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—PV

Oct. 15: The Neon Demon

Where to watch: Prime Video, Hoopla, Freevee

Modeling is a cutthroat industry, and that goes double in Drive director Nicolas Winding Refn’s 2016 horror movie The Neon Demon.

The movie follows Elle Fanning as Jesse, a young model who just moved to Los Angeles and quickly gets taken under the wing of makeup artist Ruby (Jena Malone).
As Jesse becomes more successful as a model, she meets various members of the industry who are willing to part the seas for her even as she makes enemies who
get caught up in a wake she unknowingly creates. Even when people are being nice to her, no one lets Jesse pass without making it clear they’re jealous of her
beauty and curious about what she can give them… or that they can take from her.

Here, like in Drive, Refn’s version of LA is soaked in dread and seediness in every corner.
</item 43>

<item 44>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
Divination and other occult practices are condemned by the Bible numerous times, Reehil noted.

"The very action of using [the Ouija board] has profound spiritual consequences [that are] beyond our control," he said.

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Perhaps the most famous instance of purported demonic possession connected to Ouija board usage was the case of "Roland Doe," a young teenage boy from Maryland,
who underwent numerous major exorcisms in 1949, said Reehil.

Doe's story was adapted into the 1971 book "The Exorcist" by William Peter Blatty, which was then made into the 1973 movie starring Ellen Burstyn and Linda
Blair.

Reehil told Fox News Digital that he has personally seen six cases of demonic oppression that stemmed from Ouija board usage.
</item 44>

<item 45>
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
</item 45>

<item 46>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Unlike A Nightmare on Elm Street or Friday the 13th, however, Freddy’s is rated PG-13, rather than R.

So is it really as kid-friendly as other PG-13 IP movies, like Aquaman or Captain Marvel? That will depend a lot on the kid in question, of course, but the
movie is far from an unrelenting thrill ride. It isn’t as intense as fellow PG-13 horror movies The Ring or Drag Me to Hell. The most ghoulish aspects of the
Freddy’s backstory are kept off screen or entirely unmentioned, and none of its set-pieces seem like pure nightmare fuel. Several of the titular five nights
pass without much incident at all.

That said, there are plentiful cheap music-sting jump scares and a few gruesome bits of junior-level gore. Violence-wise, this is a harder PG-13 than your
average superhero movie, which will probably only entice its younger YouTube-versed fans even more.
</item 46>

<item 47>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Rose brings Sharon to the town in a last-ditch attempt to find answers, but things go horribly wrong when the fog-shrouded town seems to transport them (along
with police officer Cybil Bennett) to a new monstrous dimension. Meanwhile, after the mother and daughter have been gone for a few hours, Rose’s husband,
Christopher, sets off to Silent Hill to find his family.

Inside the town is where Silent Hill takes the most inspiration from the video game series it’s based on. Some of Silent Hill’s most recognizable enemies and
monsters show up, chasing Cybil, Sharon, and Rose through the town and eventually into the iconic Pyramid Head. These segments are impressively scary, and
steeped in the dreadful atmosphere that made the games famous.

But even beyond the terrifying town itself, the best part of Silent Hill comes when Christopher arrives. Rather than managing to find his family, all he finds
is an abandoned mining town.
</item 47>

<item 48>
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
</item 48>

<item 49>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
The Thing follows a group of researchers working at an Antarctic base. Suddenly, a dog from a local Norwegian camp rushes into their base, with Norwegian men
hot on its heels, trying to kill it by any means necessary. However, once the American crew takes the dog in and shelters it, they discover it’s an alien that
can transform into any living creature, mimicking it perfectly — and that makes every one of them a suspect.

It’s one of the great paranoid thriller premises of all time, but it just so happens to also be filled with gross and fantastic alien gore. There’s nothing
quite like The Thing.
</item 49>

<item 50>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
The creature design in this movie is terrific, as is the cinematography and the film’s score.

Having previously been slated for a world premiere at the 2020 Tribeca Film Festival and subsequently dumped on video-on-demand in the wake of the COVID-19
pandemic, Sputnik is exactly the kind of horror movie this list was intended to spotlight: a kind of rare gem of intellectually and viscerally stimulating
horror that otherwise goes unappreciated if not given the opportunity to shine. —TE

Suspiria

Director: Dario Argento

Cast: Jessica Harper, Stefania Casini, Flavio Bucci

Where to watch: Tubi, Kanopy

One of the best-looking movies of all time with one of the best soundtracks of all time. What’s better than that?

Dario Argento’s Suspiria tells the story of Suzy Bannion, an American dancer who moves to Germany to study at the prestigious Tanz Akademie. It just so happens
that the academy is run by witches.
</item 50>

<item 51>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
There’s no comics-style backstory that demands pre-movie study.

Still, complete newcomers checking out the movie to see what the fuss is about may be confused for other reasons. The movie’s backstory is parceled out
awkwardly, in ways that are sometimes confusing. Characters’ decisions don’t always make sense. And there’s as much family melodrama as thrill-ride scares.
There isn’t much you can do about it except assure yourself that any confusion you experience in watching it isn’t just you. The movie is kind of weird, and not
because of some missing fan-only information that unlocks all its inconsistencies. The for-the-fans vibe Blum described applies more to whether the average
person will care about this story, not whether they can understand it.

Will Five Nights at Freddy’s experts enjoy the movie more?

Longtime fans hungering for an exploration or expansion of the game series’ continuing mythology should prepare themselves to see something more simplified.
</item 51>

<item 52>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
—TE

Us

Director: Jordan Peele

Cast: Lupita Nyong’o, Winston Duke, Elisabeth Moss

Where to watch: Netflix

Jordan Peele’s already a horror master just three movies into his career, but Us probably still doesn’t have the reputation it deserves. His 2019 psychological
slasher had the unfortunate fate of following up the cultural phenomenon of Get Out, so it had a hard time breaking through, in the way that sophomore projects
often do. But taken on its own terms, Us is a fantastic little horror movie with tons of atmosphere and an underground society’s worth of great scares.

The movie follows the Wilson family, whose vacation is interrupted by the arrival of a group of doppelgängers who match up with each member of the family
perfectly. The clones, it turns out, are called Tethered, and where they come from is very complicated.
</item 52>

<item 53>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
A fun monster thriller that doubles as insightful commentary on U.S. intervention, ecological disasters, and much more, The Host is a high mark in Bong’s
impressive filmography. —PV

House

Director: Nobuhiko Obayashi

Cast: Kimiko Ikegami, Miki Jinbo, Kumiko Ohba

Where to watch: Max, Criterion Channel

Few movies are as weird and excellent as Nobuhiko Obayashi’s House.

The bizarre ghost story follows a group of school girls who take a vacation to a haunted mansion in the countryside of Japan. Everything starts off well enough,
but before long the kids are being attacked by demonic gates, getting eaten by pianos, or opening portals to hell — all with visually an inventive silliness few
movies have ever matched. House isn’t all that scary, but it is weird in all the best ways, and nothing else looks or feels like it.
</item 53>

<item 54>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
I vividly remember telling friends how exciting and debauched the movie was, and their response surprised me: “Clearly you haven’t seen Possessor.”

That night, I rectified this, and dear readers: They were right. As disturbing and mystifying as I found the world of Infinity Pool, Possessor was all that and
more. And I haven’t been able to get it out of my brain since. Which is… fitting, considering the movie.

Andrea Riseborough (Mandy, To Leslie) plays an elite corporate assassin who is able to inhabit other people’s bodies, getting closer to her victims by posing as
people already in their lives. It’s a haunting premise, and one Possessor takes full advantage of, both in the moments of action and in the assassin’s
contemplation about what her identity truly even is.

Possessor is an unforgettable entry into the psychological horror canon, and it brings a little sci-fi into your Halloween celebrations. Don’t miss it. —PV

Oct.
</item 54>

<item 55>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Inspired by Żuławski’s own tumultuous divorce in 1976 and his subsequent struggles with suicidal ideation, Possession blurs the line between the
autobiographical and the phantasmagorical, with hysterical performances by Neill and Adjani that vacillate between disturbing, comical, and disquietingly
sympathetic. An inspiration for everything from Ari Aster’s Midsommar to the 2016 music video for Massive Attack’s “Voodoo in My Blood,” Possession is an
essential watch for any serious horror fan. —TE

[REC]

Directors: Jaume Balaguero, Paco Plaza

Cast: Manuela Velasco, Ferran Terraza, Martha Carbonell

Where to watch: Prime Video

One of the best and most disturbing found-footage movies ever, [REC] follows a TV reporter and camera person who follow emergency workers into an apartment
building, only to discover the dark truth inside: Some of the residents are turning into monsters.
</item 55>

<item 56>
This horror flick boasts the best opening 13 minutes of a film you’ll see this year (The Age, 2023-11-15T18:30:00+00:00)
For such a horror tragic, the first goal is always “great kills”.

Advertisement

“Every time I make a horror movie, it’s like a badge of honour for me,” says Roth. “It’s like, this is my chance to make a slasher film, you know? I grew up
with Freddie, Jason, Michael Myers, Chucky – I loved those movies so much – and in the ’90s, we had Ghostface and other slashers that were overlooked, like Mute
Witness (1995) which is one of my favourites. I just thought, well, it’s time for a new slasher. I want a new killer in town, so let’s create another one.” Roth
has been horror’s most provocative maestro for decades now, a filmmaker who finds a sinister glee in pushing the envelope – it was his film Hostel (2005), after
all, that spawned the term “torture porn”.
</item 56>

<item 57>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Every person reads like a threat, and every new competitor or friend seems to be looking for a new way to drain the life out of the people around them — until
they finally actually do it. But, like all Refn projects, what really sets The Neon Demon apart is how it looks. The griminess of the movie’s world turns
brightly lit photo shoots into shadow-filled nightmares, with people who stalk the sets like jungle predators.

Though The Neon Demon didn’t get the same acclaim as other so-called elevated horror movies from the time, looking back, it fits perfectly among the ranks of
The Witch, It Follows, and Under the Skin. Refn’s nightmarish modeling movie is dripping with cynicism, cruelty, and a creeping atmosphere, with more meanness
than almost any other movie from the era could muster. But it’s all in the pursuit of beauty, of course. —AG

Oct. 16: Project Wolf Hunting

Where to watch: Hi-Yah!
</item 57>

<item 58>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
And when they find Dracula instead of gold doubloons or whatever, they unleash hell. Overacted, canted-angled, seven-string-guitar-plucking HELL. Luckily, as
Dracula hunts down local college student Mary Van Helsing (Justine Waddell), with whom he already shares a psychic connection, Mary’s dad’s new vampire-hunter
assistant Simon (young Jonny Lee Miller!) is on his tail.

Butler, never more a heartthrob, plays Dracula in a mesmerizing, baroque manner; think Nicolas Cage doing Dracula if this year’s Renfield was emo
turn-of-the-millennium trash. And he’s constantly biting unsuspecting B-list actors to add to his army of hot people, who fight the heroes in multiple
obligatory scenes of wire-fu. Writer-director Patrick Lussier cuts it all up like it’s the video for “Freak on a Leash,” and, well, objectively, it’s quite
unlike most horror movies you can watch today!
</item 58>

<item 59>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Evans’ mesmerizing Apostle deliberately starts in roughly the same place as the classic 1973 suspense thriller The Wicker Man, with a lone man heading to a
remote Welsh island after getting a letter suggesting his sister is being held against her will there by a dangerous cult. Worming his way into the community by
pretending to be a convert, Thomas (Dan Stevens, even more feral and intense than he is in The Guest or Beauty and the Beast) inevitably discovers a lot of
nasty work going on on the island.

Evans charts his own course with Apostle, veering far from the Wicker Man mold and into much bloodier territory. But he takes advantage of some of the same
ideas: the ominous isolation of the island, the close-knit secrecy of its community, the predatory ideas that have flourished there under charismatic
leadership. Stevens is particularly terrific in this movie, lunging from one scene to the next like a humanoid wolf who can barely contain his bloodlust.
(Thomas has plenty of closely held secrets of his own.)
</item 59>

<item 60>
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
</item 60>

<item 61>
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
</item 61>

<item 62>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
The possessed-animatronics horror video game Five Nights at Freddy’s is less than a decade old, but a movie version has been in the works almost since the game
launched, which seems especially protracted for a development process. Warner Bros. bought the movie rights in 2015, and the project moved over to Blumhouse a
couple of years later. Game creator Scott Cawthon wrote and rewrote the script, while kid-horror filmmakers like Gil Kenan (Monster House) and Chris Columbus
(Gremlins, the first Harry Potter and Percy Jackson movies) signed on and jumped off the project.

Finally, under director Emma Tammi, the film is hitting theaters simultaneously with a Peacock streaming debut. The franchise’s most hardcore fans probably
consider themselves beyond ready to watch the Freddy’s film. But what about everyone else? If you’re not sure, have a look at this handy (and extremely
spoiler-light) Q&A.
</item 62>

<item 63>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
"The victims were left with night terrors, suicidal ideation, despair — and one man was impaled with chicken bones in his leg that flew across the kitchen," he
said.

"Stay away from all forms of occult practices — and stay close to God."

"Once the deliverance prayers were prayed, and the participants renounced all participation with the demons, the demonic activity ceased," he also said.

"Whenever we seek to engage demons, we move away from the One True God," said Reehil.

CLICK HERE TO GET THE FOX NEWS APP

Added the priest, "Stay away from all forms of occult practices — and stay close to God."

For more Lifestyle articles, visit www.foxnews.com/lifestyle.
</item 63>

<item 64>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Elizabeth comes with an open mind, ready to inhabit Gracie and fully understand her. And yet, across Todd Haynes’ May December, we see how nothing is as simple
as Elizabeth — or the audience — thinks. Gracie is always more than meets the eye: more emotional, more aware, more monstrous. And throughout the movie, Haynes
expertly teases out the two women’s mysteries as they each try to craft an identity around the other.

Haynes’ film certainly has callbacks to other, classic movies about selfhood. But in a way, May December feels most in line with something like The Rehearsal,
which also has a lot on its mind about how blurry the line between real life and performance can be, and particularly how traumatic it is to ask kids to behave
like adults. It’s something Melton’s Joe is tragically caught in the middle of, and something May December always has on its mind, even when it feels like it’s
about other things.
</item 64>

<item 65>
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
</item 65>

<item 66>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Lovecraft and the anti-consumerist allegories of George A. Romero. The horrors hold up over 50 years after it was first released, and with just the right
combination of inadvertent silliness and legitimate terror, Messiah of Evil is a satisfying watch and an excellent start to a month of horrors. —Toussaint Egan

Oct. 2: Ginger Snaps

Where to watch: Criterion Channel, Shudder, Peacock, Vudu, Tubi, Freevee, Plex

There’s a sizable hole in the monster movie canon: There just aren’t enough good werewolf movies! Ginger Snaps is here to fix that. And if you love Jennifer’s
Body, you’re in for a bloody treat.

Directed by future Orphan Black co-creator John Fawcett, Ginger Snaps is a delightfully gruesome story about two sisters. One of them has gotten her period for
the first time, and is turned into a werewolf shortly after. Things get messy, fast.
</item 66>

<item 67>
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
</item 67>

<item 68>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Wan put his dream (nightmare?) on screen for us all to enjoy. —Matt Patches

The Mist

Director: Frank Darabont

Cast: Thomas Jane, Marcia Gay Harden, Laurie Holden

Where to watch: Digital rental/purchase

Any fan of Stephen King worth their salt knows that the so-called king of horror has a lot of movie adaptations of his work. Few films have managed to eclipse,
let alone successfully adapt, King’s capacity for horror storytelling, with the exception of (a) Stanley Kubrick’s The Shining and (b) Frank Darabont’s The
Mist.

Darabont’s third adaptation of a Stephen King story, the film stars Thomas Jane (The Expanse) as a Hollywood poster artist living in Maine who, along with his
wife and son and the rest of his neighbors, takes shelter in a supermarket in the wake of a mysterious storm that covers the town in a deadly mist.
</item 68>

<item 69>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
In this film, a group of survivors hide out in an abandoned house in western Pennsylvania at the start of a zombie apocalypse. Led by the level-headed Ben
(Duane Jones), the group not only has to deal with the conflict of zombies trying to break in, but internal conflicts stemming from disagreements on how to
handle their precarious predicament.

Night of the Living Dead is the first example of Romero’s typical blend of jaw-dropping (and stomach-churning) practical effects and astute social commentary.
Fun fact: This movie came out a month before the MPAA film rating system, which meant a heaping amount of controversy when children were able to see the quite
graphic movie in theaters. And another fun fact: Night of the Living Dead was never copyrighted and is in the public domain because of an error by the original
theatrical distributor.
</item 69>

<item 70>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
It just so happens that the academy is run by witches. As the facade of the school unravels, Suzy’s fellow students slowly start going missing or dropping dead
in increasingly bizarre and horrible ways.

While the plot for Suspiria is interesting, what really makes the movie great is how it looks and how it sounds. Everything about the production design, the
costumes, and the colors is eccentric in ways no other horror movie has ever matched. Couple all that with the incredible and haunting soundtrack from European
rock band Goblin, and Suspiria becomes an unforgettable horror classic that everyone should see at least once. — AG

The Texas Chain Saw Massacre

Director: Tobe Hooper

Cast: Marilyn Burns, Gunnar Hansen, Allen Danziger

Where to watch: Peacock, Shudder, AMC Plus, Tubi, Freevee

Another shoestring production gone huge, Tobe Hooper’s 1974 masterpiece made over $30 million at the box office on a budget of around $140,000.
</item 70>

<item 71>
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
</item 71>

<item 72>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
And Thirst, his most overt foray into horror, may also be his steamiest movie (a high bar, considering the excellent The Handmaiden).

Frequent Park collaborator Song Kang-ho stars as a Catholic priest who volunteers for a medical experiment. The experiment instead turns him into a vampire,
giving him a lust for blood… and also for the wife of an old friend.

In many ways, this movie is in close relationship to his two most recent releases — The Handmaiden and Decision to Leave. It pairs the eroticism of the former
with the charged relationship dynamic of the latter, to gripping effect. With Park’s typical attention to detail and eye for arresting images (as well as his
wicked sense of humor), Thirst is an unconventional and sexy Halloween watch. —PV

Oct.
</item 72>

<item 73>
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
</item 73>

<item 74>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Beleaguered movie theaters got a boost this year from film fans who made an event out of seeing two radically different movies back-to-back: a slyly satirical
movie about a living fashion toy, and a heavy, grim historical drama about the father of the atomic bomb. While Barbie and Oppenheimer do have a clear theme in
common, they’re still radically different movies. But the way they collectively dominated the 2023 box office and the cultural discussion is a strong reminder
that regardless of tone or topic, a great story, told passionately, stands out. Cinephiles can be drawn to just about any movie in any genre, if it’s well
crafted and engaging, no matter what kind of fruit it is.

Here at Polygon, we have pretty eclectic collective tastes: Some of us are hungriest for action, horror, epic fantasy, or challenging science fiction, while
some of us prefer complicated drama or dark neo-noir, and others are drawn to musicals, comedy, animated adventures, and other lighter fare.

Our best-of-the-year list reflects that range.
</item 74>

<item 75>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
It is also one of the best. A pair of siblings awake in the middle of the night to find their home transformed into an inescapable nightmare of yawning
hallways, dimly lit corridors, and looming walls devoid of any windows or doors. As they search aimlessly for safety and familiarity in this otherworldly
situation, the children find that they are not alone in the house, haunted by an unknowable entity that seems to feed on their fear, as it twists their lives
into ever more frightening shapes. Or maybe not! The film’s narrative is notoriously sparse and open to interpretation, but the effect is nonetheless one of the
most unsettling scenarios seen this year. —TE

29. Are You There God? It’s Me, Margaret

Director: Kelly Fremon Craig

Cast: Abby Ryder Fortson, Rachel McAdams, Benny Safdie

Where to watch: Starz, or for digital rental/purchase

Judy Blume’s coming-of-age novel is one of the most challenged in the country, because of its direct discussion of puberty.
</item 75>

<item 76>
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
</item 76>

<item 77>
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
</item 77>

<item 78>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

If there’s one thing we love here at Polygon, it’s the Halloween season.

We cover horror year-round, whether it’s the best horror movies you can stream at home or the best horror movies on Netflix, and generally, we have our finger
on the undead pulse of the newest and ghastliest releases in horror. We even have a list of the best horror movies of the year (ranked by scariness, of course).

Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.
</item 78>

<item 79>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
We don’t dig into the considerable lore behind FNAF here — there are endless YouTube videos for that. We’re mostly talking about who this film is for and what
to expect.

Does Five Nights at Freddy’s have a post-credits scene?

Yes. More specifically, it has a brief mid-credits scene that rolls less than a minute after the initial cut to black. It isn’t integral to the movie’s lore,
and doesn’t feel like a sequel tease, though it certainly could be elaborated on in a sequel. It’s more of a comic scene involving YouTuber and horror-game
aficionado Cory DeVante “CoryxKenshin” Williams, though he doesn’t seem to be playing himself. But as far as a true post-credits scene that actually plays at
the tail end of the credit roll, Five Nights at Freddy’s doesn’t have one — just a brief, not especially exciting audio-only Easter egg.

Can you enjoy Five Nights at Freddy’s if you haven’t played the games?
</item 79>

<item 80>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Kuroneko centers on the story of two women, a mother and her daughter-in-law, who are raped and murdered by a wandering band of samurai who set fire to their
family home. Haunted by vengeance, their spirits forge a pact with the demons of the underworld, resurrecting them in exchange for the death of any samurai
unfortunate enough to cross their paths. As the death toll rises, a young samurai is enlisted to fend off the wayward spirits once and for all, unaware of
either the origins behind their malice or their tragic connection to his own past.

Kiyomi Kuroda’s cinematography in Kuroneko is a master class in minimalism, employing an intense contrast of light and shadow to produce sequences that feel
downright otherworldly to behold.
</item 80>

<item 81>
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
</item 81>

<item 82>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
—AG

Crimes of the Future

Director: David Cronenberg

Cast: Viggo Mortensen, Léa Seydoux, Kristen Stewart

Where to watch: Hulu

Microplastics: They’re everywhere!

They’re in our lungs, our blood, our food and drinking water; even the air we breathe. What the fuck is it doing to our bodies? We don’t really know, but David
Cronenberg’s 2022 body horror drama sure has an idea of what it might mean for our children. Crimes of the Future imagines a world where humans have lost the
ability to feel pain. In addition to that, several people have developed a disturbing disorder which causes their bodies to spontaneously spawn new organs.

This new reality has spawned a trend: Live surgery, wherein performance artists plagued with this condition tear into their own bodies in an effort to shape
meaning out of this strange new biological fact.
</item 82>

<item 83>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Typical enough for an adaptation, some of the more complicated lore has been stripped out, though it could probably be reintroduced in any sequel. This does
make the movie more accessible for franchise newcomers, whether Blum wanted that or not.

Also, fans might expect more screen time than they actually get for the core animatronics team of Freddy, Bonnie, Chica, Foxy, and Carl the Cupcake.

Is Five Nights at Freddy’s safe for kids?

For an ostensible horror franchise — one based on nostalgia for and subversion of animatronic characters that date to the 1980s — the Five Nights at Freddy’s
games have a surprising amount of young fans. Its kid appeal is an echo of a time when playgrounds were alight with elementary school kids describing the antics
of Freddy and/or Jason, whether witnessed firsthand or pieced together from rumors.
</item 83>

<item 84>
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
</item 84>

<item 85>
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
</item 85>

<item 86>
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
</item 86>

<item 87>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Sandman season 2

Netflix announced in November 2022 that its live-action series Sandman, based on Neil Gaiman’s seminal horror-fantasy comic, had been greenlit for a second
season. No date has been announced yet, and given the scale of the visual effects involved in the first season, it’s a fair bet that we won’t see the second
season until late 2024 or early 2025 at the earliest. Given that the first season covered the arcs of the first two Sandman graphic novels, Preludes and
Nocturnes and The Doll’s House (with side stories from the third volume, Dream Country), it’s a fair bet that season 2 will continue with book four, Season of
Mists, the heavily foreshadowed confrontation between Morpheus (Tom Sturridge) and Lucifer (Gwendoline Christie).
</item 87>

<item 88>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
Killers is a classic Scorsese Catholic reckoning. This time, the director is concerned with white supremacy and systemic racism, full of unsparing and punishing
conclusions and self recriminations in the wake of George Floyd’s death and the 2020 protests. (In a press conference on Oct. 16, Scorsese described the film as
“a story of complicity, a story of sin by omission.”)

Scorsese is expressing guilt and culpability as a beneficiary of this inequality. In the movie’s closing minutes, he literally puts himself squarely in the
frame. That’s a level of awareness that goes beyond the introverted morality plays he grappled with earlier in his career: It’s a mature framing of the
individual, and the responsibilities we have toward one another, on a societal scale.

In a charitable light, adding himself to the movie as the voice of its final reckoning is a bold statement of accountability.
</item 88>

<item 89>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
When a tech whiz (Allison Williams) has to take responsibility for her dead sister’s kid, she distances herself by building a robot to handle parenting for her.
It goes badly. Scripted by Akela Cooper with the same shamelessly messy, giddy verve she brought to Malignant, M3GAN is a good time for knowledgable horror
fans, packed with referential humor, but still channeling some real anxiety about the place computers have in our lives and how far that’ll eventually go. —TR

33. When Evil Lurks

Director: Demián Rugna

Cast: Ezequiel Rodríguez, Demián Salomón, Luis Ziembrowski

Where to watch: Shudder, AMC Plus, or for digital rental/purchase

Demián Rugna’s gory possession movie isn’t just the usual barrage of religious imagery and religious anxiety: It’s much more concerned with bigger, sadder
thoughts on the ways our institutions keep failing us, and the ways we fail each other and ourselves.
</item 89>

<item 90>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

Whether it’s something gory and macabre, silly and irreverent, or eerie and unsettling, the genre of horror is as rich and varied as the multitude of ghosts,
ghoulies, and homicidal maniacs that go bump in the night.

Looking for the best horror films available to stream on Netflix, Hulu, Max, and Paramount Plus? No worries, we’ve got the goods. We’ve combed through the
libraries of each of the major streaming platforms to bring you a list of our most recommended horror movies. Here are the best horror movies you can stream
right now, from old classics to new hits. Our latest update added Encounters of the Spooky Kind.
</item 90>

<item 91>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
The film follows a family (Jonathan Groff, Ben Aldridge, and Kristen Cui) who, after traveling to a secluded cabin in rural Pennsylvania for a vacation trip,
find themselves menaced by a group of four heavily armed intruders who refuse to let them leave. The family is presented with a warning and an ultimatum: Every
single person on the planet is about to die, and the only way to prevent this catastrophe is for one of them to choose to sacrifice themselves for the sake of
humanity. As the family struggles to escape from their captors, the undeniable weight of their situation begins to set in. Could their kidnappers be telling the
truth? And if so, would any of them be willing to submit to the unthinkable cruelty of their fate?

Knock at the Cabin is a film that challenges its audience to look inward for the answer to these questions, all the while witnessing its story unfold.
</item 91>

<item 92>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The canister is filled with some kind of liquid that defies science, and when it starts to get released, a strange evil seems to come into the world with it.

This is Carpenter at his most ambitious. While most of this movie is set in just one building, all manner of horror makes its way inside. There are people made
of bugs, horrific zombies, an extradimensional Satan, and things even worse to behold. The entire movie is an onslaught of creepy, skin-crawling images, all
built out of a unique mythology about the relationship between good and evil that underpins the world. And, most importantly, after this parade of horrors seems
to come to an end, Carpenter saves Prince of Darkness’ biggest, weirdest, most dreadful scare for last. —Austen Goslin

Oct. 8: Gravity Falls - Northwest Mansion Mystery

Where to watch: Hulu, Disney Plus (season 2, episode 10)

The more time that has passed since Gravity Falls concluded, the more miraculous it feels in hindsight.
</item 92>

<item 93>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
It’s an idea that dates as far back as the spiritualism movement of the late 19th century, and one that takes on a frightening and fascinating dimension in
Kiyoshi Kurosawa’s post-Y2K horror-thriller Pulse.

Alternating between two storylines, Kurosawa’s film follows a group of young adults and university students as a rash of inexplicable disappearances and
suicides occurs around Tokyo. As the protagonists investigate further, they slowly uncover a shocking revelation: The souls of the dead are spilling over into
the mortal world and ensnaring their unsuspecting prey through the very infrastructure of the internet. By the time they realize this, however, it’s too late;
the unearthly contagion has taken on a life of its own, and the only way to survive is cling to what few connections they have left.
</item 93>

<item 94>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Rather than managing to find his family, all he finds is an abandoned mining town. For most of the movie, the women’s frantic running and hiding from monsters
is paralleled by Christopher wandering around the exact same areas, filled with a dreadful sense that his family is close, but completely unable to see them or
the horrors threatening to kill them. It’s not a subtle metaphor for the differences in the ways that men and women go through the world, but it is an effective
one, and it manages to create a level of crushing existential terror and defeat that few movies ever come close to. —AG

Oct. 21: Suspiria (2018)

Where to watch: Prime Video, Freevee

Dario Argento’s 1977 Suspiria is a horror masterpiece, the sort of film even the best directors would face insurmountable odds of matching with a remake.
</item 94>

<item 95>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story — an American ballerina who moves to Germany to attend a prestigious ballet company, only to be greeted by a series of murders and a supernatural
mystery — is a vessel for Argento’s craft, featuring some of his boldest visuals and some truly nauseating body horror.

So how, then, did Luca Guadagnino do the impossible, creating its contemporary equal?

He cast Tilda Swinton in multiple roles and Mia Goth to do the Mia Goth thing. For cinematography, he brought on Sayombhu Mukdeeprom, known for Guadagnino’s own
Call Me By Your Name, but also Apichatpong Weerasethakul’s seminal indie films Uncle Boonmee Who Can Recall His Past Lives and Memoria. And for the score: Thom
Yorke. Yes, Radiohead Thom Yorke.
</item 95>

<item 96>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
—PR

13. Godzilla Minus One

Director: Takashi Yamazaki

Cast: Ryunosuke Kamiki, Minami Hamabe, Yuki Yamada

Where to watch: Theaters

Godzilla fans aren’t wanting for new media, including the recent Apple TV series Monarch. These recent releases, while enjoyable, have taken a distinctly
Hollywood approach, building out a mythology in the style of the MCU. Godzilla Minus One is the first Japanese live-action Godzilla film from Toho since the
brilliant, post-Fukushima Shin Godzilla. And it returns to the original recipe.

Minus One is a throwback: to the post-WWII Tokyo setting, to 1950s human melodrama, and to a kaiju singularly focused on the obliteration of large, human-made
structures.
</item 96>

<item 97>
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
</item 97>

<item 98>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
But last season really solidified the reason I even watched the show in the first place. Namely, when AHS is good, it’s really good.

AHS: NYC opens up with a serial killer on the prowl. That, coupled with the leather-clad spectral figure that seems to be haunting the main characters, seems
like it’s just typical AHS schlock. But it’s so much more than that. Yes, there is a serial killer who uncannily resembles Jeffrey Dahmer, and the scenes where
he stalks and tortures his victims are pretty scary. And yes, there is a leather daddy ghost that has no name or face. And yes, there’s a lot of dubiously
consensual BDSM scenes. But the real horror comes from the powerlessness of the queer community in the face of the AIDS crisis.

Creator Ryan Murphy builds this overwhelming, inescapable sense of dread. All AHS seasons hinge on their titular premise, which, more often than not, is a
location.
</item 98>

<item 99>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Faking her death and confining her to the grounds of his mansion, Génessier devises a desperate and horrifying scheme: Perform a series of skin grafts to repair
Christiane’s face using the faces of abducted women who resemble her. With the failure of each attempt, the body count begins to rise, prompting the authorities
to mount a desperate manhunt to apprehend the culprit. In the midst of all of this, Christiane herself grows tired of her father’s obsession, wishing instead to
reunite with her lover Jacques and resume her life.

Fantastical and atmospheric, Eyes Without a Face is a beautiful horror thriller with masterful cinematography courtesy of Eugen Schüfftan (who previously worked
on Fritz Lang’s Metropolis) and a memorable score composed by Maurice-Alexis Jarre, who would later go on to win an Academy Award for his work on Lawrence of
Arabia. If you’re looking for an intelligent and gorgeous horror movie that still manages to dabble in its share of gore, Georges Franju’s masterpiece is a
must-watch.
</item 99>

<item 100>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
And for the score: Thom Yorke. Yes, Radiohead Thom Yorke.

Where other directors would have obsessively recreated a source text, rebooting it into something unrecognizable, Guadagnino’s film is an homage — the sort of
movie you imagine Argento would make with the budgets and creative freedom afforded by tech companies trying to buy cultural capital. It’s more political. More
grotesque. The ending is just more.

The truth is 2018’s Suspiria and the original Suspiria should be enjoyed together. In the past, that’s been a challenge, with the original film being
surprisingly difficult to watch — especially if you prefer streaming. But this month, Criterion Collection has you covered, allowing for one of the most batshit
double features of your life. —CP

Oct.
</item 100>
=== END ITEM id=8748d4aafb87a2f8 ===

=== ITEM id=ab17bda3d138fbc8 ===
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

=== END ITEM id=ab17bda3d138fbc8 ===

=== ITEM id=461f29d088d8f796 ===
Gold sentences and facts:
Sentence 0: "And if the Braves let this be the story of their series, that's on them.
  Fact 0.0: A quoted speaker says that if the Braves let this be the story of their series, that is on the Braves.
Sentence 1: But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.
  Fact 1.0: Nocturne could have easily settled and coasted on classic Castlevania vibes.
  Fact 1.1: The new Nocturne show steps its game up considerably.
  Fact 1.2: The new Nocturne show uses the Castlevania franchise's iconic lore.
  Fact 1.3: The new Nocturne show uses the franchise's iconic lore to challenge the viewer's understanding of what it truly means to fight for freedom, equality, and fraternity.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Despite being without star big Aron Baynes (suspended) and import guard Shannon Scott (hamstring), the Bullets soared to a second straight victory on Sunday and
snapped a quirky streak in the process.

It was their first win after seven straight losses in Sunday fixtures, in a run that dates back to last season.

Far more importantly, the result levelled Brisbane’s current season record at 4-4 after five rounds.

But it didn’t come easily as fourth-quarter specialists Tasmania trimmed a seven-point deficit back to a single point when Majok Deng drove to the rack with 34
seconds remaining.

Advertisement

The two sides traded blows in the dying stages and the JackJumpers almost sent the game to overtime when they pinched Sam McDaniel’s inbounds pass on the final
play.

Jack McVeigh got a hand in to force the turnover and Tasmania found Jordon Crawford, but his decent look at an attempted game-leveller rimmed out.
</item 3>

<item 4>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But we held on.”

NBA and NBL champion Ian Clark top-scored with 19 points, with rookie Kyle Bowen adding a career-best 17 as did Jo Lual-Acuil along with 10 rebounds.

Advertisement

Jack McVeigh led the way for Tasmania with 20 points and seven rebounds, with Jordon Crawford contributing 16 points and four assists, and Milton Doyle 12
points and five assists.

Tasmania coach Scott Roth was philosophical afterwards, having given up 18 offensive rebounds and lost the battle on the glass by 16.

“We got a little uncharacteristic with our shot selection, which was a credit to them,” he said.

“Then they beat us up on the glass. It’s just little things when you get in those kind of games that are going to make the separation points.”
</item 4>

<item 5>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
A third straight defeat for the Wildcats has them struggling at 2-4 this season.

Coach John Rillie is focused know they aren’t playing well and the improvement needs to start offensively.

“We’re not playing good basketball at this point in time,” Rillie said.

“An area we need to improve on is our offence. That’s where we have to find some type of rhythm and we’ve yet to do that in these last couple of games.

Advertisement

“The game’s played at both ends and you need to find a rhythm to generate some natural enthusiasm, and we haven’t been able to do that for good stretches.”

Brisbane coach Justin Schueller always knew losing his superstar centre Baynes with a five-game suspension was going to be challenging, but that was magnified
on Saturday night.
</item 5>

<item 6>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The guys are going in the right direction. “I’m not disappointed, I’m stoked.”

Bullets shoot down weakened Wildcats

Advertisement

The Brisbane Bullets have shrugged off the absence of two important players to outwork the Wildcats at their home RAC Arena in Perth and produce a performance
full of heart for their 84-79 NBL win.

Both teams were 2-4 coming into Friday’s match and on losing slides, but it was the Wildcats who had the pressure building, even with the Bullets being without
key pair, the suspended Aron Baynes and injured Shannon Scott.

Brisbane scored the first seven points before opening up a 20-point lead in the third quarter and going on to win by five after holding out against a furious
late charge from the home team.

The team effort of the Bullets was outstanding as they got another 17 points from captain Nathan Sobey.
</item 6>

<item 7>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The JackJumpers opened up on fire, racing to a 17-4 lead inside the opening five minutes including three balls to Crawford and McVeigh, but United soon steadied
and it was the introduction of Lual-Acuil that inspired it.

Melbourne would close the quarter outscoring Tasmania 17 points to four before three-pointers midway through the second stanza from Bowen and Clark put them up
five.

Advertisement

United were still leading 43-42 at halftime, on the back of 11 extra rebounds and 10 second-chance points, before captain Chris Goulding opened his account at
the start of the second half from downtown.

Clark hit his second long ball of the game soon after to push Melbourne’s lead out to nine, his third stretched it to double-figures and Bowen landed two
surprise ones of his own.
</item 7>

<item 8>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Williams produced 21 points and six rebounds for South East Melbourne but Brisbane will still be without Baynes for next weekend’s double-header away to Perth
and at home to Tasmania, and he needs to find some answers.

“We’ve got to have solutions right now for two more games without Banger (Baynes) and that’s the thing that’s sitting right in the front of my head, but scoring
73 points isn’t enough to get it done in this league,” Schueller said.

“Then giving them 96 is way too many as well so we have a lot of stuff that we still have to work on.”

JLA return lifts United to win in Wollongong

Advertisement

Melbourne United have celebrated the long-awaited return of Jo Lual-Acuil but it was fellow centre Ariel Hukporti who starred in a 101-91 defeat of the
lllawarra Hawks.
</item 8>

<item 9>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Sobey and Smith then knocked down shots as the Bullets lead ballooned to 20.

The Wildcats did make a charge with a Pinder-inspired 11-0 run in the fourth to cut the deficit to three, but the Bullets answered with strong inside finishes
from Harrison and Bannan to secure the win.

United flex muscles in road win over JackJumpers

Melbourne United have had to withstand a hot start and finish from the Tasmania JackJumpers but in between were outstanding on the way to a 90-82 victory in the
NBL.

Advertisement

The league’s top two teams locked horns at a sold out MyState Bank Arena on Friday night and it was the JackJumpers who raced out of the blocks with a 17-4
start, but United took control from there.
</item 9>

<item 10>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
“When you’re missing a ton of shots you’ve got to win the offensive rebounding battle and they beat us on that.”

Tasmania burst out to a double-figure lead when they piled on 36 points during the opening period, and the margin blew out to 18 before halftime.

Doyle (12 points), Deng and Jack McVeigh (both 10) all did plenty of damage on the scoreboard in the first half for the visitors, who never trailed. Galloway’s
huge dunk sparked Sydney early in the third period and they cut the margin to three.

Advertisement

But he and Bolden soon exited and Crawford’s scintillating run gave the JackJumpers the upper hand. The Kings are hopeful Galloway will be fit for Saturday’s
trip to take on the Cairns Taipans.

“The doctors may have other plans, but he’s doing alright,” Abdelfattah said.
</item 10>
=== END ITEM id=461f29d088d8f796 ===
